# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented layer wise propagation method for convolution transpose layer, the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: Built based on the base lrp module present in lrp from Explainable AI: Interpreting, Explaining and Visualizing Deep Learning Book


import torch 
import torch.nn as nn
import torch.nn.functional as F


class ConvTranspose2DLRP(nn.ConvTranspose2d):
    """
    A subclass of `nn.ConvTranspose2d` that performs transposed convolution operations and includes additional methods for interpreting the relevance of the input data.

    Args:
        convtranspose_layer (nn.ConvTranspose2d): The convolutional layer to initialize the `ConvTranspose2DLRP` instance.

    Example:
        # Create an instance of the ConvTranspose2DLRP class
        conv_transpose = ConvTranspose2DLRP(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, dilation))

        # Copy the parameters from an existing convolutional layer
        conv_transpose.copy_parameters(existing_conv_layer)

        # Perform a forward pass on the transposed convolution layer
        output = conv_transpose.forward(input)

        # Interpret the relevance of the previous layer's input
        relevance = conv_transpose.interpret(previous_layer_input, forward_layer_relevance, rule="lrp0", parameters={})
    """

    def __init__(self, convtranspose_layer: nn.ConvTranspose2d):
        super().__init__(convtranspose_layer.in_channels,convtranspose_layer.out_channels,convtranspose_layer.kernel_size,
                         convtranspose_layer.stride,convtranspose_layer.padding,convtranspose_layer.output_padding,
                         convtranspose_layer.groups,convtranspose_layer.dilation)
        self.copy_parameters(convtranspose_layer)

    def copy_parameters(self, module):
        """
        Copies the parameters from the given module to the current instance of the class.

        Args:
            module: The module to copy the parameters from.
        """
        with torch.no_grad():
            self.weight.data.copy_(module.weight.data)
            if self.bias is not None and module.bias is not None:
                self.bias.data.copy_(module.bias.data)
            self.kernel_size = module.kernel_size
            self.stride = module.stride
            self.padding = module.padding
            self.output_padding=module.output_padding
            self.dilation = module.dilation
            self.groups = module.groups

    def forward(self,input:torch.tensor):
        """
        Performs a forward pass on the transposed convolution layer.

        Args:
            input (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor.
        """
        return super(ConvTranspose2DLRP, self).forward(input)

    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor,rule="lrp0",parameters={}) -> torch.tensor:
        """
        Calculates the relevance of the previous layer's input based on the relevance of the forward layer's output.

        Args:
            previouslayer_input (torch.tensor): The input tensor of the previous layer.
            fowardlayerrelevance (torch.tensor): The relevance tensor of the forward layer's output.
            rule (str, optional): The interpretation rule to use. Defaults to "lrp0".
            parameters (dict, optional): Additional parameters for the interpretation rule. Defaults to {}.

        Returns:
            torch.tensor: The relevance tensor of the previous layer's input.
        """
        Aij=previouslayer_input.data.requires_grad_(True)
        Zk=super(ConvTranspose2DLRP, self).forward(Aij) #equation 7.1 from chapter 7 (zk=ε + ∑0,j aj · wjk )
        sensitivity=(fowardlayerrelevance/Zk).data #equation 7.2 from chapter 7 (sk=Rk/zk)
        sensitivity[Zk==0]=0  #converted all sensitvity value whose pre-activation are zero to zero
        (Zk*sensitivity).sum().backward() # calculating the gradients (backpropagating the forward layer to present layer inputs)
        layerrelevance=(Aij*Aij.grad).data # extracting the relevance values R_j = a_j ·  c_j
        return layerrelevance
    