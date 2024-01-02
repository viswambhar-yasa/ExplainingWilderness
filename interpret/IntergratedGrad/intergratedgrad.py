# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Intergrated Gradients is calculated the gradients during backpropagation, these gradient height the area of interest, which can be used to perform prediction 
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: Based on the basic tutorial presented in https://www.tensorflow.org/tutorials/interpretability/integrated_gradients.


import torch
import torch.nn as nn
from copy import deepcopy

class IntergratedGrad(nn.Module):
    """
    A class that performs integrated gradients interpretation on a given PyTorch model.
    Used Visualizing Deep Networks by Optimizing with Integrated Gradients to understand the equations
    Args:
        model (torch.nn.Module): The PyTorch model to interpret.

    Attributes:
        model (torch.nn.Module): A deep copy of the input model.
        layer_grads (dict): A dictionary to store the gradients captured during the backward pass for specific layers.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = deepcopy(model)
        self.model.eval()
        self.layer_grads = {}

    def backward_hook_func(self, module, input_grad, output_grad):
        """
        Hook function to capture gradients during the backward pass for specific layers.

        Args:
            module: The module for which the gradients are being captured.
            input_grad: The input gradients.
            output_grad: The output gradients.
        """
        self.layer_grads[module] = input_grad[0]

    def register_backward_hook(self, model):
        """
        Registers backward hooks on the model's layers to capture gradients during the backward pass.

        Args:
            model (torch.nn.Module): The model to register backward hooks on.
        """
        for name, layer in model.named_children():
            if isinstance(layer, (nn.ReLU, nn.Flatten, nn.Identity, nn.BatchNorm2d, nn.Linear, nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Dropout)):
                layer.register_forward_hook(self.hook_function)
            else:
                self.register_backward_hook(layer)

    def integratedGradients(self, inputs, predictedclass, baseinput=None, totalsteps=50, step_size=1):
        """
        Calculates the integrated gradients for a given input and predicted class. Equation 3 

        Args:
            inputs: The input for which to calculate the integrated gradients.
            predictedclass: The predicted class index.
            baseinput: The baseline input for the integral approximation.
            totalsteps: The total number of steps for the integral approximation.
            step_size: The step size for the integral approximation.

        Returns:
            The integrated gradients for the given input and predicted class.
        """
        inputs.requires_grad_(True)
        if baseinput is None:
            baseinput = torch.zeros_like(inputs)
        attributes = 0
        for step in range(1, totalsteps,step_size):# equation 4 to solve intergation by dividing it into small intervals from input ot baseinputs
            Riemmann_approximation = baseinput + (step / totalsteps) * (inputs - baseinput)
            output = self.model(Riemmann_approximation)[:, int(predictedclass)]
            output.backward()
            attributes += inputs.grad[0] / totalsteps
        return attributes

    def forward(self, inputs):
        """
        Forward pass function that passes the inputs through the model and returns the output.

        Args:
            inputs: The input to pass through the model.

        Returns:
            The output of the model.
        """
        output = self.model(inputs)
        return output

    def interpet(self, inputs, output_type="softmax", baseinput=None, totalsteps=50, step_size=1):
        """
        Interprets the model's output by calculating attributions and integrated gradients based on the predicted class.

        Args:
            inputs: The input to interpret.
            output_type (str): The type of output to use for interpretation. Default is "softmax".
            baseinput: The baseline input for the integral approximation.
            totalsteps: The total number of steps for the integral approximation.
            step_size: The step size for the integral approximation.

        Returns:
            The interpreted attributions and integrated gradients.
        """
        output = self.model(inputs)
        if output_type == "softmax": # selecting class 
            predictions = torch.softmax(output, dim=-1)
            predictedclass, _ = torch.max(predictions, dim=1, keepdim=True)
        elif output_type == "log_softmax":
            predictions = torch.softmax(output, dim=-1)
            log_predictions = torch.log(predictions / (1 - predictions))
            predictedclass, _ = torch.max(log_predictions, dim=1, keepdim=True)
        attributes = self.integratedGradients(inputs, predictedclass, baseinput, totalsteps, step_size)
        IntergratedGrad_attributes = attributes * inputs # multiplying gradient x inputs
        return IntergratedGrad_attributes, attributes
