# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented layer wise propagation method for adaptive average pool layer, the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: Built based on the base lrp module present in lrp from Explainable AI: Interpreting, Explaining and Visualizing Deep Learning Book

import torch 
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAvgPool2dLRP(nn.AdaptiveAvgPool2d):
    """
    A subclass of `nn.AdaptiveAvgPool2d` that implements a layer-wise propagation method for the adaptive average pooling layer.

    Args:
        avgpool_layer (nn.AdaptiveAvgPool2d): The adaptive average pooling layer.

    Attributes:
        output_size (Tuple[int, int]): The output size of the adaptive average pooling layer.

    Example Usage:
        # Create an instance of the AdaptiveAvgPool2dLRP class
        avgpool_layer = nn.AdaptiveAvgPool2d((2, 2))
        adaptive_avgpool_lrp = AdaptiveAvgPool2dLRP(avgpool_layer)

        # Perform forward pass
        input = torch.randn(1, 3, 4, 4)
        output = adaptive_avgpool_lrp.forward(input)

        # Interpret the layer's relevance
        previouslayer_input = torch.randn(1, 3, 8, 8)
        fowardlayerrelevance = torch.randn(1, 3, 2, 2)
        layerrelevance = adaptive_avgpool_lrp.interpet(previouslayer_input, fowardlayerrelevance, rule="lrp0", parameters={})

    """

    def __init__(self, avgpool_layer: nn.AdaptiveAvgPool2d) -> None:
        super().__init__(avgpool_layer.output_size)
        self.output_size = avgpool_layer.output_size

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Performs the forward pass of the adaptive average pooling layer.

        Args:
            input (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor.

        """
        return super(AdaptiveAvgPool2dLRP, self).forward(input)

    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor, rule="lrp0", parameters={}) -> torch.tensor:
        """
        Interprets the relevance of the layer by calculating the layer relevance based on the input, forward layer relevance, and the specified rule and parameters.

        Args:
            previouslayer_input (torch.tensor): The input tensor of the previous layer.
            fowardlayerrelevance (torch.tensor): The relevance tensor of the forward layer.
            rule (str, optional): The rule to use for interpretation. Defaults to "lrp0".
            parameters (dict, optional): Additional parameters for interpretation. Defaults to {}.

        Returns:
            torch.tensor: The relevance tensor of the layer.

        """
        Aij = previouslayer_input.data.requires_grad_(True)
        if rule == "lrpzplus" or rule == "lrpepsilon":
            Zk = super(AdaptiveAvgPool2dLRP, self).forward(Aij) #equation 7.1 from chapter 7 (zk=ε + ∑0,j aj · wjk )
            fowardlayerrelevance = fowardlayerrelevance.view(Zk.shape)
            if rule == "lrpzplus":
                Zk = Zk.clamp(min=0) #clamping the negative value based on the propagation rule
            Zk += torch.sign(Zk) * parameters["epsilon"] # implementing epsilon rule 

            sensitivity = (fowardlayerrelevance / Zk).data #equation 7.2 from chapter 7 (sk=Rk/zk)
            sensitivity[Zk == 0] = 0 
            (Zk * sensitivity).sum().backward() # calculating the gradients (backpropagating the forward layer value to present layer inputs)
            layerrelevance = (Aij * Aij.grad).data # extracting the relevance values R_j = a_j ·  c_j
            return layerrelevance
        else: #init type based on the pseudo code from Explainable AI: Interpreting, Explaining and Visualizing Deep Learning Book
            Zk = super(AdaptiveAvgPool2dLRP, self).forward(Aij)
            fowardlayerrelevance = fowardlayerrelevance.view(Zk.shape)
            sensitivity = (fowardlayerrelevance / Zk).data
            sensitivity[Zk == 0] = 0
            (Zk * sensitivity).sum().backward()
            layerrelevance = (Aij * Aij.grad).data
            return layerrelevance
        