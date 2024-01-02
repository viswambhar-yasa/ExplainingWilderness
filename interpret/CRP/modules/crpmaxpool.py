# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented concept relevance propagation method for max pooling layer., the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 


import torch 
import torch.nn as nn

class MaxPool2dCRP(nn.MaxPool2d):
    """
    A subclass of `nn.MaxPool2d` that extends the functionality of the MaxPool2d layer by adding additional methods for interpreting the relevance of input features.

    Args:
        maxpool_layer (torch.nn.MaxPool2d): The MaxPool2d layer to be extended.

    Attributes:
        return_indices (bool): A boolean indicating whether to return the indices of the maximum values in the forward pass.
        kernel_size (Union[int, Tuple[int, int]]): The size of the kernel used for pooling.
        stride (Union[int, Tuple[int, int]]): The stride of the pooling operation.
        padding (Union[int, Tuple[int, int]]): The padding added to the input.
        indicies (Optional[torch.Tensor]): The indices of the maximum values in the forward pass.

    Methods:
        __init__(self, maxpool_layer: torch.nn.MaxPool2d) -> None:
            Initializes the MaxPool2dCRP instance by calling the constructor of the parent class nn.MaxPool2d and copying the parameters of the maxpool_layer instance.

        copy_parameters(self, module):
            Copies the kernel size, stride, and padding parameters from the given module to the current instance.

        forward(self, input: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the MaxPool2dCRP layer by calling the forward method of the parent class nn.MaxPool2d and returns the output tensor.

        interpret(self, previouslayer_input: torch.Tensor, forwardlayerrelevance: dict, concepts=None, conceptindex_estimation=None, top_num=2, rule="lrp0", parameters={}) -> torch.Tensor:
            Interprets the relevance of input features by calculating the relevance of each concept in the forwardlayerrelevance dictionary. It performs operations such as clamping, adding epsilon, and max unpooling to calculate the relevance. The relevance of each concept is returned as a dictionary.
    """

    def __init__(self, maxpool_layer: torch.nn.MaxPool2d) -> None:
        super().__init__(maxpool_layer.kernel_size, maxpool_layer.stride, maxpool_layer.padding, maxpool_layer.dilation, maxpool_layer.return_indices, maxpool_layer.ceil_mode)
        self.copy_parameters(maxpool_layer)
        self.return_indices = True

    def copy_parameters(self, module):
        """
        Copies the kernel size, stride, and padding parameters from the given module to the current instance.

        Args:
            module: The module from which to copy the parameters.
        """
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the MaxPool2dCRP layer by calling the forward method of the parent class nn.MaxPool2d and returns the output tensor.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x, self.indicies = super(MaxPool2dCRP, self).forward(input)
        return x
    
    def interpet(self, previouslayer_input: torch.Tensor, forwardlayerrelevance: dict, concepts=None, conceptindex_estimation=None, top_num=2, rule="lrp0", parameters={}) -> torch.Tensor:
        """
        interpet the relevance of input features by calculating the relevance of each concept in the forwardlayerrelevance dictionary. It performs operations such as clamping, adding epsilon, and max unpooling to calculate the relevance. The relevance of each concept is returned as a dictionary.

        Args:
            previouslayer_input (torch.Tensor): The input tensor of the previous layer.
            forwardlayerrelevance (dict): The relevance of each concept in the forward layer.
            concepts (Optional): Not used.
            conceptindex_estimation (Optional): Not used.
            top_num (int): The number of top concepts to consider.
            rule (str): The rule to use for relevance interpretation.
            parameters (dict): Additional parameters for relevance interpretation.

        Returns:
            torch.Tensor: The relevance of each concept in the previous layer.
        """
        Aij = previouslayer_input.data.requires_grad_(True) 
        outputconceptrelevance = {} 
        Zk, _ = super(MaxPool2dCRP, self).forward(Aij)
        outputsize = Aij.shape
        
        if rule == "zplus" or rule == "epsilon":
            Zk = Zk.clamp(min=0)
            Zk += torch.sign(Zk) * parameters["epsilon"] 
        
        for index, conceptrelevance in forwardlayerrelevance.items(): #Algorithm 2 Concept Relevance Propagation page-46 chapter 7
            conceptrelevance = conceptrelevance.view(Zk.shape)
            sensitivity = (conceptrelevance / Zk)
            sensitivity[Zk == 0] = 0# for max pooling the indices are stored and unpooled to their original position
            R = torch.nn.functional.max_unpool2d(sensitivity, self.indicies, self.kernel_size, self.stride, self.padding, outputsize)
            layerrelevance = (R * Aij)
            outputconceptrelevance[index] = layerrelevance
                
        return outputconceptrelevance
        