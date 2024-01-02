# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented concept relevance propagation class, the class converts the model to CRP layer modules and perform .
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import torch
import torch.nn as nn

from interpret.CRP.modules.crplinear import LinearCRP
from interpret.CRP.modules.crpconv2d import Conv2DCRP
from interpret.CRP.modules.crpmaxpool import MaxPool2dCRP
from interpret.CRP.modules.crpadaptivepool import AdaptiveAvgPool2dCRP

class CRPModel(nn.Module):
    """
    The `CRPModel` class is a subclass of `nn.Module` that wraps a given model and adds additional functionality for interpreting the relevance of input features and concepts.

    Attributes:
        model (torch.nn.Module): The wrapped model.
        layer_inputs (dict): A dictionary that stores the inputs of each layer during forward pass.
        layer_relevance (list): A list that stores the relevance values of each layer during relevance interpretation.
        layer_length (int): The number of layers in the model.
        lrp_default_parameters (dict): A dictionary that stores the default parameters for each interpretation rule.

    Methods:
        __init__(self, model: torch.nn.Module) -> None: Initializes the `CRPModel` object by wrapping the given model and converting its layers to CRP versions.
        converter(self, model): Converts the layers of the given model to their corresponding CRP versions recursively.
        hook_function(self, module, input, output): Hook function to capture the inputs of each layer during forward pass.
        recursive_forward_hook(self, model, index): Recursively registers hooks to capture the inputs of each layer during forward pass.
        forward(self, input: torch.tensor) -> torch.tensor: Performs the forward pass of the model and returns the output.
        interpret(self, input, output_type="max", rule="lrpepsilon", concept_ids={}, parameter={}, estimate_relevance={}, concept_type="relmax", top_num=2, input_zbetaplus=False): Performs relevance interpretation on the input using the specified interpretation rule and returns the interpreted relevance values.
    """
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model=model
        self.model.eval()
        self.converter(self.model) # converts the model with CRP modules
        self.layer_inputs={}
        self.layer_relevance=[]
        self.layer_length=0
        print("Index of the layer required for concept registration /n")
        self.recursive_foward_hook(self.model,self.layer_length)
        #default rule parameter
        self.lrp_default_parameters={
            "lrp0":{},
        "epsilon": {"epsilon": 1, "gamma": 0},
        "gamma": {"epsilon": 0.25, "gamma": 0.1},
        "alpha1beta0": {"alpha": 1, "beta": 0,"epsilon": 1e-2, "gamma": 0},
        "zplus": {"epsilon": 1e-2},
        "alphabeta": {"epsilon": 1, "gamma": 0, "alpha": 2, "beta": 1}}
        
    def converter(self, model):
        """
        Recursively converts the layers of the given model to their corresponding CRP (Concept Relevance Propagation) versions.

        Args:
            model (torch.nn.Module): The model to be converted to CRP versions.

        Returns:
            None. The method modifies the given model in-place by replacing its layers with their CRP versions.
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(model, name, LinearCRP(module))
            elif isinstance(module, nn.Conv2d):
                setattr(model, name, Conv2DCRP(module))
            elif isinstance(module, nn.MaxPool2d):
                setattr(model, name, MaxPool2dCRP(module))
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                setattr(model, name, AdaptiveAvgPool2dCRP(module))
            else:
                self.converter(module)
            

    def hook_function(self, module, input, output):
        """
        Store the inputs of each layer during the forward pass of the model.

        Args:
            module (nn.Module): The layer module for which the inputs need to be captured.
            input (Tensor): The input tensor to the layer.
            output (Tensor): The output tensor from the layer.

        Returns:
            None. The method simply stores the inputs of each layer in the `layer_inputs` dictionary.
        """
        self.layer_inputs[module] = input[0]

    def recursive_foward_hook(self, model, index):
        """
        Recursively registers forward hooks on the layers of a given model.

        Args:
            model (torch.nn.Module): The model for which the forward hooks need to be registered.
            index (int): The current index of the layer being processed.
        """
        for name, layer in model.named_children():
            if isinstance(layer, (LinearCRP, Conv2DCRP, MaxPool2dCRP, AdaptiveAvgPool2dCRP)):
                print(self.layer_length, layer)
                layer.register_forward_hook(self.hook_function)
                self.layer_length += 1
            else:
                if isinstance(layer, (nn.ReLU, nn.Flatten, nn.Identity, nn.BatchNorm2d, nn.MaxPool2d)):
                    continue
                else:
                    self.recursive_foward_hook(layer, self.layer_length)
            


    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass of the model.

        Args:
            input (torch.tensor): The input tensor to be passed through the model.

        Returns:
            torch.tensor: The output tensor obtained from the forward pass of the model.
        """
        return self.model(input)
    

    def interpet(self, input, output_type="max", rule="lrpepsilon", concept_ids={}, parameter={}, estimate_relevance={}, concept_type="relmax", top_num=2, input_zbetaplus=False):
        """
        Perform relevance interpretation on the input using a specified interpretation rule.

        Args:
            input (torch.Tensor): The input tensor to be passed through the model.
            output_type (str, optional): The type of output to be used for relevance interpretation. Can be either "softmax" or "max". Defaults to "max".
            rule (str, optional): The interpretation rule to be used for relevance interpretation. Defaults to "lrpepsilon".
            concept_ids (dict, optional): A dictionary that maps layer indices to concept IDs for relevance estimation. Defaults to {}.
            parameter (dict, optional): A dictionary that stores the parameters for the interpretation rule. Defaults to {}.
            estimate_relevance (dict, optional): A dictionary that specifies the number of top concepts to estimate relevance for each layer. Defaults to {}.
            concept_type (str, optional): The type of concept estimation to be used. Defaults to "relmax".
            top_num (int, optional): The number of top concepts to consider for relevance estimation. Defaults to 2.
            input_zbetaplus (bool, optional): A boolean flag indicating whether to use a specific interpretation rule for the last layer. Defaults to False.

        Returns:
            dict: A dictionary containing the interpreted relevance values for each layer and concept.
        """
        self.layer_relevance = []
        output = self.model(input)
        # output relevance is the important factor which influences the propagation rule
        
        if output_type == "softmax":#softmax method, the softmax value of the max class is propagted as the inital value (if the value is lower, the relevance values is lower)
            relevance = torch.softmax(output, dim=-1)
            dict_relevance = {"": relevance}
        elif output_type == "max": # identifying the max class and assigning 1 to it(100% of relevance is given as input)
            predictions = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(predictions, dim=1, keepdim=True)
            mask = (predictions == max_values)
            relevance = predictions * mask
            relevance = (relevance > 0).float()
            relevance[relevance == 0] = -1
            dict_relevance = {"": relevance}
        else:
            dict_relevance = {"": output}
        #appending concept relevance of each layer
        self.layer_relevance.append(dict_relevance)
        if not parameter:#selecting the parameters
            parameter = self.lrp_default_parameters[rule]
        Nlayer = len(list(self.layer_inputs.items())) - 1
        if "all" in estimate_relevance:
            estimate_concept_all_layer = True
            top_num = estimate_relevance["all"]
        estimate_concept_all_layer = False
        #looping over each layer and performing concept relevance propagation 
        for index, (layer, layer_input) in enumerate(reversed(self.layer_inputs.items())):
            if isinstance(layer, LinearCRP) or isinstance(layer, Conv2DCRP) or isinstance(layer, AdaptiveAvgPool2dCRP) or isinstance(layer, MaxPool2dCRP):
                if index == Nlayer and input_zbetaplus:
                    rule = "zbetalh"
                    parameter = self.lrp_default_parameters["epsilon"]
                concept_index = (Nlayer - index)
                if concept_index in concept_ids: #extracting individual concepts from the input dictionary
                    concepts = concept_ids[concept_index]
                else:
                    concepts = None
                if estimate_concept_all_layer:# for all channels within the layer
                    dict_relevance = layer.interpet(layer_input, dict_relevance, concepts, concept_type, top_num, rule, parameter)
                else:
                    if concept_index in estimate_relevance:
                        conceptestimation_type = concept_type
                        top_num = estimate_relevance[concept_index]#number of top concepts for each layer
                    else:
                        conceptestimation_type = None
                    dict_relevance = layer.interpet(layer_input, dict_relevance, concepts, conceptestimation_type, top_num, rule, parameter)
        return dict_relevance