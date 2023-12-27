# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented concept relevance propagation method for adaptive average pooling layer., the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import torch 
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAvgPool2dCRP(nn.AdaptiveAvgPool2d):
    """
    A subclass of `nn.AdaptiveAvgPool2d` that performs adaptive average pooling and interprets the relevance of the previous layer's input.

    Args:
        avgpool_layer (nn.AdaptiveAvgPool2d): The adaptive average pooling layer.

    Attributes:
        output_size (Tuple[int, int]): The output size of the adaptive average pooling layer.

    Methods:
        forward(input: torch.Tensor) -> torch.Tensor:
            Performs the forward pass of the adaptive average pooling operation on the input tensor and returns the output tensor.

        interpret(previouslayer_input: torch.Tensor, forwardlayerrelevance: dict, concepts=None, conceptindex_estimation=None, top_num=2, rule="lrp0", parameters={}) -> dict:
            Interprets the relevance of the previous layer's input based on the relevance of the forward layer's output. It calculates the relevance using the LRP-Z+ or LRP-epsilon rule and returns a dictionary of relevance tensors for each concept.
    """

    def __init__(self, avgpool_layer: nn.AdaptiveAvgPool2d) -> None:
        """
        Initializes the AdaptiveAvgPool2dCRP class.

        Args:
            avgpool_layer (nn.AdaptiveAvgPool2d): The adaptive average pooling layer.
        """
        super().__init__(avgpool_layer.output_size)
        self.output_size = avgpool_layer.output_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the adaptive average pooling operation on the input tensor and returns the output tensor.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return super(AdaptiveAvgPool2dCRP, self).forward(input)
    
    def interpret(self, previouslayer_input: torch.Tensor, forwardlayerrelevance: dict, concepts=None, conceptindex_estimation=None, top_num=2, rule="lrp0", parameters={}) -> dict:
        """
        Interprets the relevance of the previous layer's input based on the relevance of the forward layer's output.

        Args:
            previouslayer_input (torch.Tensor): The input tensor of the previous layer.
            forwardlayerrelevance (dict): A dictionary of relevance tensors for each concept from the forward layer.
            concepts (optional): The concepts to interpret. Defaults to None.
            conceptindex_estimation (optional): The estimated concept indices. Defaults to None.
            top_num (int): The number of top concepts to consider. Defaults to 2.
            rule (str): The rule to use for relevance calculation. Defaults to "lrp0".
            parameters (dict): Additional parameters for relevance calculation. Defaults to {}.

        Returns:
            dict: A dictionary of relevance tensors for each concept.
        """
        Aij = previouslayer_input.data.requires_grad_(True)
        outputconceptrelevance = {}
        Zk = super(AdaptiveAvgPool2dCRP, self).forward(Aij)
        n, c, h, w = Aij.shape
        n, C, H, W = Zk.shape
        #calculating parameters for backpropagation 
        stride = (int(h/H), int(w/W))
        kernel_size = (h-(H-1)*stride[0], (w-(W-1)*stride[1]))
        weight = torch.ones(size=(c, C, kernel_size[0], kernel_size[1])) / (kernel_size[0] * kernel_size[1])
        if rule == "lrpzplus" or rule == "lrpepsilon": 
            if rule == "lrpzplus":
                Zk = Zk.clamp(min=0)
            Zk += torch.sign(Zk) * parameters["epsilon"]
        for index, conceptrelevance in forwardlayerrelevance.items():c
            #in adaptive pooling, the pre-activation are not needed to calculated as it is just averaged over 
            conceptrelevance = conceptrelevance.view(Zk.shape)
            sensitivity = (conceptrelevance / Zk)
            sensitivity[Zk == 0] = 0
            pR = F.conv_transpose2d(sensitivity, weight, None, stride=stride, padding=0)
            layerrelevance = pR * Aij
            outputconceptrelevance[index] = layerrelevance
        return outputconceptrelevance
            
        