# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Contain important function required for calculating global refernce images, propagation rule
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import os
import torch 
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from crp.cache import Cache
from typing import List, Tuple
from crp.concepts import Concept
from crp.statistics import Statistics
from zennit.composites import Composite
from typing import Callable, Dict, Tuple
from crp.maximization import Maximization
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.torchvision import VGGCanonizer,ResNetCanonizer
from zennit.composites import EpsilonAlpha2Beta1Flat,EpsilonPlus,EpsilonPlusFlat,EpsilonAlpha2Beta1,Epsilon


from interpret.concept.conceptcomposite import EpsilonComposite,module_map



def get_layer_types(model):
    """
    Get a dictionary of layer names and their corresponding types from a PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model.

    Returns:
    dict: A dictionary where keys are layer names and values are layer types.
    """
    layer_types = {}
    for name, module in model.named_modules():
        if name:  # This skips the top level module which is the entire model
            layer_types[name] = type(module).__name__
    return layer_types



def get_canonizer(canonizer_type):
    """
    Returns a list of canonizers based on the given `canonizer_type`.

    Args:
        canonizer_type (str): The type of canonizer to be returned. Valid options are "vgg16", "resnet18", or any other value for a custom canonizer.

    Returns:
        list: A list containing the requested canonizer instance.
    """
    if canonizer_type=="vgg16":
        return [VGGCanonizer()]
    elif canonizer_type=="resnet18":
        return [ResNetCanonizer()]
    else:
        return [SequentialMergeBatchNorm()]


def get_composite(compositename, canonizername, layer_map=None):
    """
    Returns a composite object based on the given composite name and canonizer name.

    Args:
        compositename (str): The name of the composite. Valid options are "epsilonplus", "epsilonalphabeta", "epsilonplusflat", "epsilonalphabetaflat", "epsilon", or "layerspecific".
        canonizername (str): The name of the canonizer. Valid options are "vgg16", "resnet18", or any other value for a custom canonizer.
        layer_map (function, optional): A function that maps layers to their corresponding composite layers. Defaults to None.

    Returns:
        object: An instance of the requested composite object.
    """
    canonizer = get_canonizer(canonizername)
    if compositename == "epsilonplus":
        return EpsilonPlus(canonizers=canonizer)
    elif compositename == "epsilonalphabeta":
        return EpsilonAlpha2Beta1(canonizers=canonizer)
    elif compositename == "epsilonplusflat":
        return EpsilonPlusFlat(canonizers=canonizer)
    elif compositename == "epsilonalphabetaflat":
        return EpsilonAlpha2Beta1Flat(canonizers=canonizer)
    elif compositename == "epsilon":
        return EpsilonComposite(canonizers=canonizer)
    elif compositename == "layerspecific":
        return Composite(module_map=module_map, canonizers=canonizer)
    else:
        return Epsilon()

def get_relevance_function(output_type):
    """
    Returns a relevance function based on the specified `output_type`.

    Args:
        output_type (str): The type of relevance function to be returned. Valid options are "softmax", "max", "log_softmax", or "max_activation".

    Returns:
        function: The relevance function based on the specified `output_type`.

    Example Usage:
        output_type = "softmax"
        relevance_function = get_relevance_function(output_type)
        output = torch.tensor([0.1, 0.5, 0.4])
        relevance = relevance_function(output)
        print(relevance)

    Expected Output:
        tensor([0., 1., 0.])
    """

    def softmax_relevance(output):
        relevance = torch.softmax(output, dim=-1)
        max_values, _ = torch.max(relevance, dim=1, keepdim=True)
        mask = (relevance == max_values)
        init_relevance = (-(mask == 0).float() + (mask > 0).float())
        return init_relevance
    
    def sigmoidmax_relevance(output,threshold=0.5):
        relevance = torch.sigmoid(output)
        mask = (relevance >= threshold)
        init_relevance = mask.float()
        return init_relevance
    
    def sigmoid_relevance(output):
        init_relevance = torch.sigmoid(output)
        return init_relevance

    def max_relevance(output):
        predictions = torch.softmax(output, dim=-1)
        max_values, _ = torch.max(predictions, dim=1, keepdim=True)
        mask = (predictions == max_values)
        relevance = predictions * mask
        init_relevance = (relevance > 0).float()
        # relevance[relevance==0]=-1
        return init_relevance

    def log_softmax_relevance(output):
        relevance = torch.softmax(output, dim=-1)
        init_relevance = torch.log(relevance / (1 - relevance))
        return init_relevance
    
    def max_activation(output):
        max_values, _ = torch.max(output, dim=1, keepdim=True)
        mask = (output == max_values)
        prediction_mask = output * mask
        classes=output.shape[-1]
        init_relevance=((-output*(prediction_mask==0).float())/(classes-1))+(output*(prediction_mask>0).float())
        return init_relevance

    # Define other relevance functions here
    if output_type == "softmax":
        return softmax_relevance
    elif output_type == "max":
        return max_relevance
    elif output_type == "log_softmax":
        return log_softmax_relevance
    elif output_type == "max_activation":
        return max_activation
    elif output_type == "sigmoidmax":
        return sigmoidmax_relevance
    elif output_type == "sigmoid":
        return sigmoid_relevance
    else:
        return None
        



class ConceptRelevanceAttribute(CondAttribution):
    """
    A subclass of CondAttribution used for calculating relevance scores for concepts in a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model used for relevance calculation.
        device (torch.device, optional): The device (CPU or GPU) used for computation. Defaults to None.
        overwrite_data_grad (bool, optional): A flag indicating whether to overwrite the gradients of the input data. Defaults to True.
        no_param_grad (bool, optional): A flag indicating whether to disable gradient computation for model parameters. Defaults to True.
    """

    def __init__(self, model: nn.Module, device=None, overwrite_data_grad=True, no_param_grad=True) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model, device, overwrite_data_grad, no_param_grad)

    def relevance_init(self, prediction, target_list, init_rel):
        """
        Initializes the relevance scores based on the given prediction, target list, and initialization method.
        We inherited the base class, to have more flexibity when selecting the relevance output. By modifying the input it can be extended to work for segmentation tasks.
        Args:
            prediction (torch.Tensor): The prediction tensor.
            target_list (list): A list of target indices.
            init_rel (callable or torch.Tensor or int or np.integer): The initialization method for relevance scores.

        Returns:
            torch.Tensor: The initialized relevance scores.
        """
        if callable(init_rel):
            output_selection = init_rel(prediction)
        elif isinstance(init_rel, torch.Tensor):
            output_selection = init_rel
        elif isinstance(init_rel, (int, np.integer)):
            output_selection = torch.full(prediction.shape, init_rel)
        else:
            output_selection = prediction

        if target_list:
            mask = torch.zeros_like(output_selection)
            #print(mask)
            for i, targets in enumerate(target_list):
                mask[i, targets] = output_selection[i, targets]
            output_selection = mask
        #print(output_selection)
        return output_selection
    


class ConceptMaximization(Maximization):
    """
    A class for performing concept maximization in a neural network model. 

    Args:
        mode (str): The mode of concept maximization, either "relevance" or "activation".
        max_target (str): The target for maximization, either "sum" or "max".
        abs_norm (bool): A flag indicating whether to perform absolute normalization.
        path (str): The path to save the results of concept maximization.

    Attributes:
        mode (str): The mode of concept maximization, either "relevance" or "activation".
        max_target (str): The target for maximization, either "sum" or "max".
        abs_norm (bool): A flag indicating whether to perform absolute normalization.
        path (str): The path to save the results of concept maximization.

    Methods:
        _save_results(d_index: Tuple[int, int] = None) -> List[str]: Saves the results of concept maximization to files.
        run(composite: Composite, data_start=0, data_end=1000, batch_size=16, checkpoint=250, on_device=None): 
            Runs concept maximization on a composite object and collects the results.
    """

    def __init__(self, mode="relevance", max_target="sum", abs_norm=False, path=None):
        super().__init__(mode, max_target, abs_norm, path)

    def _save_results(self, d_index: Tuple[int, int] = None) -> List[str]:
        """
        taken from "zennit-crp"
        Saves the results of concept maximization to files. We have used the same format as in crp but made chnage in the saving 

        Args:
            d_index (Tuple[int, int], optional): The index of the concept to save. Defaults to None.

        Returns:
            List[str]: A list of the saved file paths.
        """
        saved_files = []

        for layer_name in self.d_c_sorted:
            if d_index:
                filename = f"{layer_name}_{d_index[0]}_{d_index[1]}_"
            else:
                filename = f"{layer_name}_"

            if '\\' in filename:
                filename = filename.rsplit('\\', 1)[-1]

            np.save(self.PATH / Path(filename + "data.npy"), self.d_c_sorted[layer_name].cpu().numpy())
            np.save(self.PATH / Path(filename + "rf.npy"), self.rf_c_sorted[layer_name].cpu().numpy())
            np.save(self.PATH / Path(filename + "rel.npy"), self.rel_c_sorted[layer_name].cpu().numpy())

            saved_files.append(str(self.PATH / Path(filename)))

        self.delete_result_arrays()
        return saved_files



class ConceptStatistics(Statistics):
    """
    A class for collecting and organizing the results of concept maximization.

    Args:
        mode (str): The mode of concept maximization, either "relevance" or "activation".
        max_target (str): The target for maximization, either "sum" or "max".
        abs_norm (bool): A flag indicating whether to perform absolute normalization.
        path (str): The path to save the results of concept maximization.

    Example Usage:
        concept_stats = ConceptStatistics(mode="relevance", max_target="sum", abs_norm=False, path="/results/")
        path_list = ["/results/concept1/", "/results/concept2/"]
        concept_stats.collect_results(path_list)
        concept_stats.save_results()
    """

    def __init__(self, mode="relevance", max_target="sum", abs_norm=False, path=None):
        super().__init__(mode, max_target, abs_norm, path)

    def collect_results(self, path_list: List[str], d_index: Tuple[int, int] = None):
        """
        taken from "zennit-crp"
        Collects the results of concept maximization from the specified paths and organizes them into result arrays.

        Args:
            path_list (List[str]): A list of paths where the results are stored.
            d_index (Tuple[int, int], optional): The index of the result array to be saved. Defaults to None.

        Returns:
            List[str]: A list of file paths where the results are saved.
        """
        self.delete_result_arrays()

        pbar = tqdm(total=len(path_list), dynamic_ncols=True)

        for path in path_list:

            l_name, filename = path.split("/")[-2:]
            target = filename.split("_")[0]

            d_c_sorted = np.load(path + "data.npy")
            rf_c_sorted = np.load(path + "rf.npy")
            rel_c_sorted = np.load(path + "rel.npy")

            d_c_sorted, rf_c_sorted, rel_c_sorted = map(torch.from_numpy, [d_c_sorted, rf_c_sorted, rel_c_sorted])

            self.concatenate_with_results(l_name, target, d_c_sorted, rel_c_sorted, rf_c_sorted)
            self.sort_result_array(l_name, target)

            pbar.update(1)

        for path in path_list:
            for suffix in ["data.npy", "rf.npy", "rel.npy"]:
                os.remove(path + suffix)

        pbar.close()

        return self._save_results(d_index)

    def _save_results(self, d_index: Tuple[int, int] = None) -> List[str]:
        """
        Saves the results of concept maximization to files.

        Args:
            d_index (Tuple[int, int], optional): The index of the result array to be saved. Defaults to None.

        Returns:
            List[str]: A list of file paths where the results are saved.
        """
        return super()._save_results(d_index)





class ConceptVisualization(FeatureVisualization):
    """
    A class for performing concept visualization in a neural network model. We have used the custom class and create a custom concept visualization class

    Args:
        attribution (CondAttribution): An instance of the `CondAttribution` class.
        dataset: The dataset used for concept visualization.
        layer_map (Dict[str, Concept]): A dictionary mapping layer names to `Concept` objects.
        preprocess_fn (function, optional): A function for preprocessing the input data. Defaults to None.
        max_target (str, optional): The target for maximization, either "sum" or "max". Defaults to "sum".
        abs_norm (bool, optional): A flag indicating whether to perform absolute normalization. Defaults to True.
        path (str, optional): The path to save the results of concept visualization. Defaults to "ConceptVisualization".
        device (str, optional): The device to run the visualization on (e.g., "cuda" or "cpu"). Defaults to None.
        cache (Cache, optional): An instance of the `Cache` class. Defaults to None.

    Returns:
        list: A list of file paths where the results of concept visualization are saved.
    """
    def __init__(self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept], preprocess_fn = None, max_target="sum", abs_norm=True, path="ConceptVisualization", device=None, cache: Cache = None):
        super().__init__(attribution, dataset, layer_map, preprocess_fn, max_target, abs_norm, path, device, cache)
        self.RelMax = ConceptMaximization("relevance", max_target, abs_norm, path)
        self.ActMax = ConceptMaximization("activation", max_target, abs_norm, path)

        self.RelStats = Statistics("relevance", max_target, abs_norm, path)
        self.ActStats = Statistics("activation", max_target, abs_norm, path)

    def run(self, composite: Composite, data_start=0, data_end=1000, batch_size=16, checkpoint=250, on_device=None):
        """
        Runs concept visualization on a given composite object.

        Args:
            composite (Composite): The composite object to perform concept visualization on.
            data_start (int, optional): The starting index of the data to visualize. Defaults to 0.
            data_end (int, optional): The ending index of the data to visualize. Defaults to 1000.
            batch_size (int, optional): The batch size for visualization. Defaults to 16.
            checkpoint (int, optional): The checkpoint interval for saving results. Defaults to 250.
            on_device (str, optional): The device to run the visualization on. Defaults to None.

        Returns:
            list: A list of file paths where the results of concept visualization are saved.
        """
        print("Running Analysis...")
        saved_checkpoints = self.run_distributed(composite, data_start, data_end, batch_size, checkpoint, on_device)

        print("Collecting concepts...")
        saved_files = self.collect_results(saved_checkpoints)
        return saved_files
    
