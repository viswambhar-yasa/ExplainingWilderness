# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented layer wise propagation class, the class converts the model to LRP layer modules and perform .
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: The model layers is converted to have additionals functionalities which can be used for forward and backward propagation along with explaination. 

import torch
import torch.nn as nn
from interpret.LRP.modules.lrplinear import LinearLRP
from interpret.LRP.modules.lrpconv2d import Conv2DLRP
from interpret.LRP.modules.lrpmaxpool import MaxPool2dLRP
from interpret.LRP.modules.lrpadaptivepool import AdaptiveAvgPool2dLRP


class LRPModel(nn.Module):
    """
    The `LRPModel` class is a subclass of `nn.Module` in PyTorch. It is used for interpreting the relevance of input features in a neural network model using different LRP (Layer-wise Relevance Propagation) rules.
    
    Example Usage:
        model = LRPModel(my_model)
        input = torch.tensor(...)
        relevance = model.interpet(input, output_type="max", rule="lrpzplus")
    
    Main functionalities:
    - Converting linear, convolutional, max pooling, and adaptive average pooling layers in the model to their LRP counterparts (`LinearLRP`, `Conv2DLRP`, `MaxPool2dLRP`, `AdaptiveAvgPool2dLRP`).
    - Registering forward hooks to capture the input tensors of the converted layers.
    - Performing forward propagation and relevance interpretation using the converted layers and LRP rules.
    - Providing default LRP parameters for different LRP rules.
    
    Methods:
    - `__init__(self, model: torch.nn.Module)`: Initializes the `LRPModel` object by setting the model, converting its layers, and registering forward hooks.
    - `converter(self, model)`: Converts linear, convolutional, max pooling, and adaptive average pooling layers in the model to their LRP counterparts.
    - `hook_function(self, module, input, output)`: Stores the input tensor of a layer in the `layer_inputs` dictionary.
    - `recursive_foward_hook(self, model)`: Recursively registers forward hooks for the layers in the model.
    - `forward(self, input: torch.tensor)`: Performs forward propagation using the model.
    - `interpet(self, input, output_type="max", rule="lrpzplus", parameter={}, input_zbetaplus=True)`: Interprets the relevance of input features based on the chosen LRP rule and returns the relevance tensor.
    
    Fields:
    - `model`: The original model passed to the `LRPModel` constructor.
    - `layer_inputs`: A dictionary that stores the input tensors of the converted layers.
    - `layer_relevance`: A list that stores the relevance tensors of the converted layers.
    - `lrp_default_parameters`: A dictionary that provides default LRP parameters for different LRP rules.
    """
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model=model
        self.model.eval()
        self.converter(self.model)
        self.layer_inputs={}
        self.layer_relevance=[]
        self.recursive_foward_hook(self.model)
        #default hyperparameter required for different propagation rule
        self.lrp_default_parameters={
            "lrp0":{},
        "lrpepsilon": {"epsilon": 1, "gamma": 0},
        "lrpgamma": {"epsilon": 0.25, "gamma": 0.1},
        "lrpalpha1beta0": {"alpha": 1, "beta": 0,"epsilon": 1e-2, "gamma": 0},
        "lrpzplus": {"epsilon": 1e-2},
        "lrpalphabeta": {"epsilon": 1, "gamma": 0, "alpha": 2, "beta": 1}}
        
    def converter(self, model):
        """
        Converts linear, convolutional, max pooling, and adaptive average pooling layers in a given model to their corresponding LRP (Layer-wise Relevance Propagation) counterparts.

        Args:
            model (nn.Module): The original model that needs to be converted to LRP model.

        Returns:
            None

        Example:
            model = LRPModel(my_model)
            model.converter(model)
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(model, name, LinearLRP(module))
            elif isinstance(module, nn.Conv2d):
                setattr(model, name, Conv2DLRP(module))
            elif isinstance(module, nn.MaxPool2d):
                setattr(model, name, MaxPool2dLRP(module))
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                setattr(model, name, AdaptiveAvgPool2dLRP(module))
            else:
                self.converter(module)


    def hook_function(self, module, input, output):
        """
        Store the input tensor of a layer in the layer_inputs dictionary , during forward propagation.

        Args:
            module (nn.Module): The layer module for which the input tensor needs to be stored.
            input (Tensor): The input tensor of the layer.
            output (Tensor): The output tensor of the layer.

        Returns:
            None
        """
        self.layer_inputs[module] = input[0]

    def recursive_foward_hook(self, model):
        """
        Recursively registers forward hooks for specific layers in the given model.

        Args:
            model (nn.Module): The model for which forward hooks need to be registered.

        Returns:
            None
        """
        for name, layer in model.named_children(): # conveert the based layer to LRP layer modules with changing it's forward or backward functionailty
            # if name=="downsampe":
            #     continue
            if isinstance(layer, (LinearLRP, Conv2DLRP, MaxPool2dLRP, AdaptiveAvgPool2dLRP)):
                layer.register_forward_hook(self.hook_function)
            else:
                if isinstance(layer, (nn.ReLU, nn.Flatten, nn.Identity, nn.BatchNorm2d)):
                    continue
                else:
                    self.recursive_foward_hook(layer)


    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Perform forward propagation using the model.

        Args:
            input (torch.tensor): The input tensor to be passed through the model.

        Returns:
            torch.tensor: The output tensor of the model.
        """
        return self.model(input)
    
    def interpet(self, input, output_type="max", rule="lrpzplus", parameter={}, input_zbetaplus=True):
        """
        Interpret the relevance of input features in a neural network model using different LRP (Layer-wise Relevance Propagation) rules.

        Args:
            input (torch.Tensor): The input tensor to be passed through the model.
            output_type (str, optional): The type of output to be used for relevance interpretation. Default is "max".
            rule (str, optional): The LRP rule to be used for relevance interpretation. Default is "lrpzplus".
            parameter (dict, optional): Additional parameters for the LRP rule. Default is an empty dictionary.
            input_zbetaplus (bool, optional): A boolean value indicating whether to use the LRP rule "lrpzbetalh" for the last layer. Default is True.

        Returns:
            torch.Tensor: The relevance tensor, which represents the interpreted relevance of the input features based on the chosen LRP rule.
        """
        self.layer_relevance = []
        output = self.model(input)
        # output relevance is the important factor which influences the propagation rule
        if output_type == "softmax": #softmax method, the softmax value of the max class is propagted as the inital value (if the value is lower, the relevance values is lower)
            relevance = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(relevance, dim=1, keepdim=True)
            mask = (relevance == max_values)
            relevance *= (-(mask == 0).float() + (mask > 0).float())
        elif output_type == "max": # identifying the max class and assigning 1 to it(100% of relevance is given as input)
            predictions = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(predictions, dim=1, keepdim=True)
            mask = (predictions == max_values)
            relevance = predictions * mask
            relevance = (relevance > 0).float()
        elif output_type == "log_softmax": # used in alpha beta propagation rule to show the influence of the alpha and beta value 
            relevance = torch.softmax(output, dim=-1)
            relevance = torch.log(relevance / (1 - relevance))
        elif output_type == "softmax_grad":
            predictions = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(output, dim=-1, keepdim=True)
            mask = (output == max_values)
            prediction_mask = predictions * mask
            relevance = prediction_mask + ((predictions * output) * (mask == 0).float())
        elif output_type == "max_activation": # classes with max activation are filtered, to identiy features that are positive and get the top 5 classes with same features
            max_values, _ = torch.max(output, dim=1, keepdim=True)
            mask = (output == max_values)
            prediction_mask = output * mask
            classes = output.shape[-1]
            relevance = ((-output * (prediction_mask == 0).float()) / (classes - 1)) + (output * (prediction_mask > 0).float())
        else:
            if (output.shape[1] == 1):
                relevance = (output > 0.5).float()
                relevance[relevance == 0] = -1
            else:
                relevance = torch.zeros_like(output)
                relevance[:, 0] = 1

        self.layer_relevance.append(relevance) #container to store the relevance values of each layer, require to perform relevance analysis
        if not parameter:
            parameter = self.lrp_default_parameters[rule] #if not parameters are provided, default values are selected.
        Nlayer = len(list(self.layer_inputs.items())) - 1
        for index, (layer, layer_input) in enumerate(reversed(self.layer_inputs.items())): # running the back propagation rule
            if isinstance(layer, LinearLRP) or isinstance(layer, Conv2DLRP) or isinstance(layer, AdaptiveAvgPool2dLRP) or isinstance(layer, MaxPool2dLRP):
                if index == Nlayer and input_zbetaplus:
                    rule = "lrpzbetalh"
                    parameter = self.lrp_default_parameters["lrpepsilon"]
                relevance = layer.interpet(layer_input, relevance, rule, parameter)
                self.layer_relevance.append(relevance)
        return relevance
