# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented layer wise propagation method for maxpool layer, the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: Built based on the base lrp module present in lrp from Explainable AI: Interpreting, Explaining and Visualizing Deep Learning Book

import torch 
import torch.nn as nn

class MaxPool2dLRP(nn.MaxPool2d):
    """
    A subclass of `torch.nn.MaxPool2d` that implements a modified version of the LRP (Layer-wise Relevance Propagation) algorithm for interpreting and explaining deep learning models.
    """

    def __init__(self, maxpool_layer: torch.nn.MaxPool2d) -> None:
        """
        Initializes the MaxPool2dLRP class.

        Args:
            maxpool_layer (torch.nn.MaxPool2d): The max pooling layer to be used.

        """
        super().__init__(maxpool_layer.kernel_size, maxpool_layer.stride, maxpool_layer.padding, maxpool_layer.dilation, maxpool_layer.return_indices, maxpool_layer.ceil_mode)
        self.copy_parameters(maxpool_layer)

    def copy_parameters(self, module):
        """
        Copies the parameters from the given module to the current module.

        Args:
            module: The module from which to copy the parameters.

        """
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
    

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Performs the max pooling operation on the input tensor and returns the result.

        Args:
            input (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The result of the max pooling operation.

        """
        return super(MaxPool2dLRP, self).forward(input)
    
    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor, rule="lrp0", parameters={}) -> torch.tensor:
        """
        Computes the relevance of the previous layer's input based on the relevance of the forward layer's output.

        Args:
            previouslayer_input (torch.tensor): The input tensor of the previous layer.
            fowardlayerrelevance (torch.tensor): The relevance tensor of the forward layer's output.
            rule (str, optional): The LRP rule to be used (default is "lrp0").
            parameters (dict, optional): Additional parameters that can be used by specific LRP rules.

        Returns:
            torch.tensor: The relevance tensor of the previous layer's input.

        """
        Aij = previouslayer_input.data.requires_grad_(True)  
        if rule == "lrpzplus" or rule == "lrpepsilon":
            Zk = super(MaxPool2dLRP, self).forward(Aij) #equation 7.1 from chapter 7 (zk=ε + ∑0,j aj · wjk )
            fowardlayerrelevance = fowardlayerrelevance.view(Zk.shape)
            if rule == "lrpzplus":
                Zk = Zk.clamp(min=0) #selecting only positive pre-activation values for zplus propagation rule
            Zk += torch.sign(Zk) * parameters["epsilon"]  #equation 7.1 from chapter 7 adding epsilon
            sensitivity = (fowardlayerrelevance / Zk).data #equation 7.2 from chapter 7 (sk=Rk/zk)
            sensitivity[Zk == 0] = 0 #converted all sensitvity value whose pre-activation are zero to zero
            (Zk * sensitivity).sum().backward() # calculating the gradients (backpropagating the forward layer to present layer inputs)
            layerrelevance = (Aij * Aij.grad).data  # extracting the relevance values R_j = a_j ·  c_j
            return layerrelevance 
        else:#default mode init propagation
            Zk = super(MaxPool2dLRP, self).forward(Aij)
            fowardlayerrelevance=fowardlayerrelevance.view(Zk.shape)
            sensitivity=(fowardlayerrelevance/Zk).data
            sensitivity[Zk==0]=0
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data #R_j = a_j ·  c_j
            return layerrelevance 