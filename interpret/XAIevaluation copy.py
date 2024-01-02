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
    def __init__(self,config) -> None:
        self.config=config
        if config["device"] is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=torch.device(config["device"])
        self.model=buildmodel(model_type=config["modeltype"],multiclass_channels=config["n_classes"],modelweightpath=config["modelweightspath"]).to(self.device)
        self.availablecategories=quantus.helpers.constants.available_categories()
        self.availablemetric=quantus.helpers.constants.available_metrics()
        self.xaimethodslist=["IntergratedGradients","GradientShap","GuidedGradCam","LRP","CRP","Occulsion"]
        csvfilepath=os.path.join(config["root_dir"],config["datasplitfilename"])
        self.dataset = SentinelDataset(csvfilepath, config["root_dir"],output_channels=config["n_classes"],datasettype=config["datasettype"],filter_label=config["filterclass"],device=config["device"])
        self.layer_name=None
    def set_evaluationmetric(self,custom_metric=None):
        if custom_metric is None:
            self.metrics = {"Robustness": quantus.AvgSensitivity(nr_samples=10,lower_bound=0.2,norm_numerator=quantus.norm_func.fro_norm,norm_denominator=quantus.norm_func.fro_norm,perturb_func=quantus.perturb_func.uniform_noise,similarity_func=quantus.similarity_func.difference,abs=False,normalise=False,aggregate_func=np.mean,return_aggregate=True,disable_warnings=True,),
            "Faithfulness": quantus.FaithfulnessCorrelation(nr_runs=10,subset_size=self.config["subset_size"],perturb_baseline="black",perturb_func=quantus.perturb_func.baseline_replacement_by_indices,similarity_func=quantus.similarity_func.correlation_pearson,abs=False,normalise=False,aggregate_func=np.mean,return_aggregate=True,disable_warnings=True),
            "Complexity": quantus.Sparseness(abs=True,normalise=False,aggregate_func=np.mean,return_aggregate=True,disable_warnings=True),
            "Randomisation": quantus.RandomLogit(num_classes=self.config["n_classes"],similarity_func=quantus.similarity_func.ssim,abs=True,normalise=False,aggregate_func=np.mean,return_aggregate=True,disable_warnings=True)}
        else:
            self.metrics=custom_metric

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
        """Wrapper aorund captum's Integrated Gradients implementation.
        built based on the tutorial from Quantus on how to wrap custom methods

        Args:
            model (_type_): _description_
            inputs (_type_): _description_
            targets (_type_): _description_
            abs (bool, optional): _description_. Defaults to False.
            normalise (bool, optional): _description_. Defaults to False.

        Returns:
            np.array: _description_
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
        elif xaimethod=="Occulsion":
            heatmaps = Occlusion(self.model).attribute(inputs=inputs, target=targets, strides=kwargs.get("strides",(3,25,25)), sliding_window_shapes=kwargs.get("sliding_window_shapes",(3,50,50))).sum(axis=1).reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        elif xaimethod=="LRP":
            targets = torch.nn.functional.one_hot(targets, num_classes=self.config["n_classes"]).long().to(kwargs.get("device", None))
            composite=get_composite(kwargs.get("compositetype","epsilon"),self.config["modeltype"])
            with Gradient(model=model, composite=composite) as attributor:
                out, relevance = attributor(inputs, targets)
            heatmaps =relevance.sum(axis=1).reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        elif xaimethod=="CRP":
            Concepts=ConceptRelevance(model,device=self.config["device"])
            condition=[{"y":[i for i in range(0,self.config["n_classes"])]}]
            relevance,_,_=Concepts.conditional_relevance(inputs,condition=condition,compositename=kwargs.get("compositetype","epsilon"),canonizerstype=self.config["modeltype"],output_type="softmax",record_layer=[list(Concepts.layer_map.keys())[0]])
            heatmaps =relevance.reshape(-1, kwargs.get("img_size", 256), kwargs.get("img_size", 256)).cpu().data
        if normalise:
            heatmaps = quantus.normalise_func.normalise_by_negative(heatmaps)
        gc.collect()
        torch.cuda.empty_cache()
        if isinstance(heatmaps, torch.Tensor):
            if heatmaps.requires_grad:
                return heatmaps.cpu().detach().numpy()
            return heatmaps.cpu().numpy()

        return heatmaps

    def runevaluation(self,inputs=None,targets=None,xaimethodslist=["IntergratedGradients","GradSHAP","GuidedGradCam","LRP","CRP","Occlusion"],metricsdict=None,stop_step=100):
        self.set_evaluationmetric(metricsdict)
        if inputs is None:
            dataloader=self.build_dataset()
            dataloadelen=len(dataloader)
        if xaimethodslist is None:
            xaimethodslist=self.xaimethodslist
        evaluationresults={method:{} for method in xaimethodslist}
        for xaimethod in list(evaluationresults.keys()):
            for metric, evaluationfunc in self.metrics.items():
                print(f"Evaluating {metric} of {xaimethod} method.")
                gc.collect()
                torch.cuda.empty_cache()
                if inputs is None:
                    scorelist=[]
                    with tqdm(dataloader, unit="batch") as t:
                        for step, (images, labels) in enumerate(t):
                            if step==stop_step:
                                break
                            
                            batchscore=evaluationfunc(model=self.model,x_batch=images.detach().cpu().numpy(),y_batch=labels.detach().cpu().numpy(),a_batch=None,s_batch=None,device=self.device,explain_func=self.explainationfunction,explain_func_kwargs={"method": xaimethod,"posterior_mean": copy.deepcopy(self.model.state_dict()),"mean": 1.0,"std": 0.5,"sg_mean": 0.0,"sg_std": 0.5,"n": 25,"m": 25,"noise_type": "multiplicative","device": self.device},)
                            scorelist.append(batchscore[0])
                            #except:
                            #    batchscore=[0]
                            t.set_description(f"Step [{step+1}/{stop_step}] - Score: {batchscore[0]}")
                    metric_score=np.mean(np.array(scorelist))
                else:
                    metric_score=evaluationfunc(model=self.model,x_batch=inputs.detach().cpu().numpy(),y_batch=targets.detach().cpu().numpy(),a_batch=None,s_batch=None,device=self.device,explain_func=self.explainationfunction,explain_func_kwargs={"method": xaimethod,"posterior_mean": copy.deepcopy(self.model.state_dict()),"mean": 1.0,"std": 0.5,"sg_mean": 0.0,"sg_std": 0.5,"n": 25,"m": 25,"noise_type": "multiplicative","device": self.device,},)
                evaluationresults[xaimethod][metric] = metric_score
                print(metric_score)
            print(evaluationresults)
        # Empty cache.
            gc.collect()
            torch.cuda.empty_cache()
        return evaluationresults
    

"""
{'IntergratedGradients': {'Robustness': 1.4968426752835513, 'Faithfulness': 0.041997872918350485, 'Complexity': 0.5742761883277926, 'Randomisation': 0.35910768181308106}, 'GradientShap': {'Robustness': 1.5793331075459718, 'Faithfulness': -0.03600548244098872, 'Complexity': 0.5710438364616324, 'Randomisation': 0.38489310005651645}, 'GuidedGradCam': {'Robustness': 0.9614801134914159, 'Faithfulness': -0.04633399059349962, 'Complexity': 0.7219163604758863, 'Randomisation': 0.8313507221737124}, 'LRP': {'Robustness': 2.217645118944347, 'Faithfulness': -0.03268743740911735, 'Complexity': 0.5269719107775679, 'Randomisation': 0.9471694028504608}, 'CRP': {'Robustness': 7.436603741906584, 'Faithfulness': -0.00969879379736506, 'Complexity': 0.548771122233842, 'Randomisation': 1.0000000010825114}, 'Occulsion': {}}

{'IntergratedGradients': {'Robustness': 1.3391776829259472, 'Faithfulness': 0.3860379649138446, 'Complexity': 0.6040301678191227, 'Randomisation': 0.9958897417934376}, 'GradientShap': {'Robustness': 1.416151441563852, 'Faithfulness': 0.3601432169391239, 'Complexity': 0.6098857645759603, 'Randomisation': 0.9060430438024754}, 'GuidedGradCam': {'Robustness': 1.0399533453164622, 'Faithfulness': 0.03389993379296694, 'Complexity': 0.7561072377362782, 'Randomisation': 0.7357100289106643}, 'LRP': {'Robustness': 5.247414029890205, 'Faithfulness': -0.041882592429828336, 'Complexity': 0.5848558382199267, 'Randomisation': 0.9989472110161037}, 'CRP': {'Robustness': 7.097758830711246, 'Faithfulness': -0.052121300791054585, 'Complexity': 0.5854508989216735, 'Randomisation': 0.9999908999076453}, 'Occulsion': {'Robustness': 1.2151306146231946, 'Faithfulness': 0.02660373097400706, 'Complexity': 0.3390578147034611, 'Randomisation': 0.9999908999076453}}

"""