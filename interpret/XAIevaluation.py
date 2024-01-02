# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Using Quantus library we are calculating important explainable AI metrics
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: The quantus metric parameters and wrapper function are used based on the tutorials

import os
import gc
import torch
import copy
import quantus
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from zennit.attribution import Gradient
from sentinelmodels.pretrained_models import buildmodel
from sentinelmodels.preprocessing import SentinelDataset
from interpret.concept.conceptpropagation import get_composite
from interpret.concept.conceptrelevance import ConceptRelevance
from captum.attr import IntegratedGradients,GradientShap,GuidedGradCam,Occlusion

class XAIEvaluation:
    def __init__(self, config):
        """
        Initializes an instance of the XAIEvaluation class.

        Args:
            config (dict): A dictionary containing the configuration parameters for the XAIEvaluation class.
                - device (str): The device to be used for computation. If None, it will be set to "cuda" if available, else "cpu".
                - modeltype (str): The type of model to be used.
                - n_classes (int): The number of classes in the model.
                - modelweightspath (str): The path to the model weights.
                - root_dir (str): The root directory.
                - datasplitfilename (str): The filename of the data split CSV file.
                - datasettype (str): The type of dataset.
                - filterclass (str): The filter class.
                - batchsize (int): The batch size.
                - subset_size (int): The subset size.

        Returns:
            None
        """
        self.config = config
        if config["device"] is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config["device"])
        self.model = buildmodel(model_type=config["modeltype"], multiclass_channels=config["n_classes"], modelweightpath=config["modelweightspath"]).to(self.device)
        self.availablecategories = quantus.helpers.constants.available_categories()
        self.availablemetric = quantus.helpers.constants.available_metrics()
        self.xaimethodslist = ["IntergratedGradients", "GradientShap", "GuidedGradCam", "LRP", "CRP", "Occulsion"]
        csvfilepath = os.path.join(config["root_dir"], config["datasplitfilename"])
        self.dataset = SentinelDataset(csvfilepath, config["root_dir"], output_channels=config["n_classes"], datasettype=config["datasettype"], filter_label=config["filterclass"], device=config["device"])
        self.layer_name = None


    def set_evaluationmetric(self, custom_metric=None):
        """
        Initializes the metrics used for evaluating the performance of the XAIEvaluation class.
    
        Args:
            custom_metric (dict, optional): A dictionary containing custom metrics for evaluating the XAIEvaluation class.
    
        Returns:
            None
    
        Summary:
        The `set_evaluationmetric` method initializes the metrics used for evaluating the performance of the XAIEvaluation class. 
        It allows for custom metrics to be provided, otherwise it sets default metrics for Robustness, Faithfulness, Complexity, and Randomisation.
    
        Example Usage:
        ```python
        config = {
            "device": "cuda",
            "modeltype": "resnet",
            "n_classes": 10,
            "modelweightspath": "path/to/model/weights",
            "root_dir": "path/to/root/directory",
            "datasplitfilename": "data_split.csv",
            "datasettype": "image",
            "filterclass": "class1",
            "batchsize": 32,
            "subset_size": 1000
        }
    
        xaievaluation = XAIEvaluation(config)
        xaievaluation.set_evaluationmetric()
        ```
    
        Code Analysis:
        - If `custom_metric` is None, the method initializes default metrics for Robustness, Faithfulness, Complexity, and Randomisation.
        - Each metric is initialized with specific parameters and added to the `self.metrics` dictionary.
        #build based on the qunatus tutorial 
        """
        if custom_metric is None:
            self.metrics = {
                "Robustness": quantus.AvgSensitivity(
                    nr_samples=10,
                    lower_bound=0.2,
                    norm_numerator=quantus.norm_func.fro_norm,
                    norm_denominator=quantus.norm_func.fro_norm,
                    perturb_func=quantus.perturb_func.uniform_noise,
                    similarity_func=quantus.similarity_func.difference,
                    abs=False,
                    normalise=False,
                    aggregate_func=np.mean,
                    return_aggregate=True,
                    disable_warnings=True,
                ),
                "Faithfulness": quantus.FaithfulnessCorrelation(
                    nr_runs=10,
                    subset_size=self.config["subset_size"],
                    perturb_baseline="black",
                    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                    similarity_func=quantus.similarity_func.correlation_pearson,
                    abs=False,
                    normalise=False,
                    aggregate_func=np.mean,
                    return_aggregate=True,
                    disable_warnings=True,
                ),
                "Complexity": quantus.Sparseness(
                    abs=True,
                    normalise=False,
                    aggregate_func=np.mean,
                    return_aggregate=True,
                    disable_warnings=True,
                ),
                "Randomisation": quantus.RandomLogit(
                    num_classes=self.config["n_classes"],
                    similarity_func=quantus.similarity_func.ssim,
                    abs=True,
                    normalise=False,
                    aggregate_func=np.mean,
                    return_aggregate=True,
                    disable_warnings=True,
                ),
            }
        else:
            self.metrics = custom_metric

    def extract_first_layer_weights(self,model):
        """
        Recursive helper function to traverse nested structures and extract the first layer of a given model.

        Args:
            model (nn.Module): The input model for which the first layer needs to be extracted.

        Returns:
            nn.Module: The first layer of the given model. If no first layer is found, None is returned.
        """
        def find_first_layer(module):
            """
            Recursive function to find the first layer in a nested module.

            Args:
                module (nn.Module): The module to search for the first layer.

            Returns:
                nn.Module: The first layer found in the nested modules. If no first layer is found, None is returned.
            """
            for layer in module.children():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    return layer
                else:
                    # Recursively search in the nested modules
                    result = find_first_layer(layer)
                    if result is not None:
                        return result
            return None

      # Call the recursive helper function
        return find_first_layer(model)
    
    def build_dataset(self):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config["batchsize"], shuffle=True)
        return data_loader
    
    
    def explainationfunction(self,model, inputs, targets,abs=False, normalise=False, *args, **kwargs) -> np.array:
        """
        Wrapper around captum's attribution and custom methods to compute heatmaps. Build based on the tutorials from quantus

        Args:
            model (torch.nn.Module): The model to be used for attribution.
            inputs (torch.Tensor or array-like): The input data for which attribution is to be computed.
            targets (torch.Tensor or array-like): The target labels for the input data.
            abs (bool, optional): Whether to take the absolute value of the heatmaps. Defaults to False.
            normalise (bool, optional): Whether to normalize the heatmaps. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            np.array: The attribution heatmaps generated by the selected attribution method.
        """
        gc.collect()
        torch.cuda.empty_cache()
        # Set model in evaluate mode.
        model.to(kwargs.get("device", None))
        model.eval()

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs).to(kwargs.get("device", None))
        xaimethod=kwargs.get("method", "IntegratedGradients")
        assert (len(np.shape(inputs)) == 4), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets).long().to(kwargs.get("device", None))
        baselines = torch.zeros_like(inputs).to(kwargs.get("device", None))
        if xaimethod=="IntergratedGradients":
            heatmaps = IntegratedGradients(model).attribute(inputs=inputs, target=targets,
                                                            baselines=baselines,n_steps=10,
                                                            method="riemann_trapezoid").sum(axis=1).reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        elif xaimethod=="GradientShap":
            heatmaps = GradientShap(model).attribute(inputs=inputs, target=targets,
                                                            baselines=baselines,n_samples=10).sum(axis=1).reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        elif xaimethod=="GuidedGradCam":
            if self.layer_name is None:
                self.layer_name=self.extract_first_layer_weights(self.model)
            heatmaps = GuidedGradCam(self.model,layer=self.layer_name).attribute(inputs=inputs, target=targets).sum(axis=1).reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        elif xaimethod=="Occlusion":
            heatmaps = Occlusion(self.model).attribute(inputs=inputs, target=targets, strides=kwargs.get("strides",(3,25,25)), sliding_window_shapes=kwargs.get("sliding_window_shapes",(3,50,50))).sum(axis=1).reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        elif xaimethod=="LRP":
            targets = torch.nn.functional.one_hot(targets, num_classes=self.config["n_classes"]).long().to(kwargs.get("device", None))
            composite=get_composite(kwargs.get("compositetype","epsilon"),self.config["modeltype"])
            with Gradient(model=model, composite=composite) as attributor:
                out, relevance = attributor(inputs, targets)
            heatmaps =relevance.sum(axis=1).reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        elif xaimethod=="CRP":
            Concepts=ConceptRelevance(model,device=self.config["device"])
            condition=[{"y":[0]},{"y":[1]}]
            relevance,_,_=Concepts.conditional_relevance(inputs,condition=condition,compositename=kwargs.get("compositetype","epsilon"),canonizerstype=self.config["modeltype"],output_type="max",record_layer=list(Concepts.layer_map.keys()))
            reshaped_tensor = relevance.view(int(relevance.shape[0]/2), 2, relevance.shape[-2], relevance.shape[-1])
            heatmaps =reshaped_tensor.sum(axis=1).reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        if normalise:
            heatmaps = quantus.normalise_func.normalise_by_negative(heatmaps)
        gc.collect()
        torch.cuda.empty_cache()
        if isinstance(heatmaps, torch.Tensor):
            if heatmaps.requires_grad:
                return heatmaps.cpu().detach().numpy()
            return heatmaps.cpu().numpy()

        return heatmaps

    def runevaluation(self, xaimethodslist=["IntergratedGradients","GradSHAP","GuidedGradCam","LRP","CRP","Occlusion"], metricsdict=None):
        """
        Evaluates the performance of different XAI methods.

        Args:
            xaimethodslist (list, optional): A list of XAI methods to be evaluated. If not provided, the default list of XAI methods from the `XAIEvaluation` class will be used.
            metricsdict (dict, optional): A dictionary of custom metrics for evaluating the XAI methods. If not provided, default metrics for Robustness, Faithfulness, Complexity, and Randomisation will be used.

        Returns:
            dict: A dictionary containing the evaluation results for each XAI method and metric.
        """
        self.set_evaluationmetric(metricsdict)
        dataloader = self.build_dataset()
        dataloadelen = len(dataloader)
        if xaimethodslist is None:
            xaimethodslist = self.xaimethodslist
        evaluationresults = {method: {} for method in xaimethodslist}
        for xaimethod in list(evaluationresults.keys()):
            for metric, evaluationfunc in self.metrics.items():
                print(f"Evaluating {metric} of {xaimethod} method.")
                gc.collect()
                torch.cuda.empty_cache()
                scorelist = []
                with tqdm(dataloader, unit="batch") as t:
                    for step, (images, labels) in enumerate(t):
                        batchscore = evaluationfunc(model=self.model, x_batch=images.detach().cpu().numpy(), y_batch=labels.detach().cpu().numpy(), a_batch=None, s_batch=None, device=self.device, explain_func=self.explainationfunction, explain_func_kwargs={"method": xaimethod, "posterior_mean": copy.deepcopy(self.model.state_dict()), "mean": 1.0, "std": 0.5, "sg_mean": 0.0, "sg_std": 0.5, "n": 25, "m": 25, "noise_type": "multiplicative", "device": self.device},)
                        scorelist.append(batchscore[0])
                        t.set_description(f"Step [{step+1}/{dataloadelen}] - Score: {batchscore[0]}")
                metric_score = np.mean(np.array(scorelist))
                evaluationresults[xaimethod][metric] = metric_score
                print("metric ", metric, " score:", metric_score)
            print(evaluationresults[xaimethod][metric])
        # Empty cache.
        gc.collect()
        torch.cuda.empty_cache()
        return evaluationresults
    
