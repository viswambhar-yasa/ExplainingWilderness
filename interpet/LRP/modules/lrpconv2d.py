# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented layer wise propagation method for convolution layer, the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: Built based on the base lrp module present in lrp from Explainable AI: Interpreting, Explaining and Visualizing Deep Learning Book

import torch 
import torch.nn as nn
import torch.nn.functional as F

class Conv2DLRP(nn.Conv2d):
    """
    The `Conv2DLRP` class is a subclass of `nn.Conv2d` in the PyTorch library. It is used for interpreting the relevance of input features in a convolutional neural network (CNN) using different LRP (Layer-wise Relevance Propagation) rules.

    Example Usage:
        # Create an instance of the Conv2DLRP class
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        conv_lrp = Conv2DLRP(conv_layer)

        # Forward pass through the convolutional layer
        output = conv_lrp.forward(input)

        # Interpret the relevance of the previous layer's input features
        relevance = conv_lrp.interpet(previouslayer_input, fowardlayerrelevance, rule="lrp0", parameters={})

    Main functionalities:
    - Copying the parameters of a given convolutional layer to initialize the Conv2DLRP instance.
    - Forward pass through the convolutional layer.
    - Interpreting the relevance of the previous layer's input features using different LRP rules.

    Methods:
    - __init__(self, conv_layer: nn.Conv2d): Initializes the Conv2DLRP instance by copying the parameters of the given convolutional layer.
    - copy_parameters(self, module): Copies the parameters (weights and biases) of the given module to the Conv2DLRP instance.
    - forward(self, input: torch.tensor) -> torch.tensor: Performs a forward pass through the convolutional layer.
    - interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor, rule="lrp0", parameters={}) -> torch.tensor: Interprets the relevance of the previous layer's input features using different LRP rules.

    Fields:
    - weight: The weight tensor of the convolutional layer.
    - bias: The bias tensor of the convolutional layer.
    - kernel_size: The size of the convolutional kernel.
    - stride: The stride of the convolution operation.
    - padding: The padding applied to the input.
    - dilation: The dilation rate of the convolution operation.
    - groups: The number of groups used for grouped convolution.
    - padding_mode: The padding mode used for the convolution operation.
    """
    def __init__(self, conv_layer: nn.Conv2d):
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, conv_layer.dilation, conv_layer.groups, conv_layer.bias is not None, conv_layer.padding_mode)
        self.copy_parameters(conv_layer)

    def copy_parameters(self, module):
        """
        Copies the parameters (weights, biases, kernel size, stride, padding, dilation, and groups) of a given module to the Conv2DLRP instance.

        Args:
            module (nn.Module): The module from which the parameters will be copied.

        Returns:
            None
        """
        with torch.no_grad():
            self.weight.data.copy_(module.weight.data)
            if self.bias is not None and module.bias is not None:
                self.bias.data.copy_(module.bias.data)
            self.kernel_size = module.kernel_size
            self.stride = module.stride
            self.padding = module.padding
            self.dilation = module.dilation
            self.groups = module.groups

    def forward(self,input:torch.tensor):
        #print(input.shape,super(Conv2DLRP, self).forward(input).shape)
        return super(Conv2DLRP, self).forward(input)
    
    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor, rule="lrp0", parameters={}) -> torch.tensor:
        """
        Interpret the relevance of the previous layer's input features in a convolutional neural network (CNN) using different LRP (Layer-wise Relevance Propagation) rules.

        Args:
            previouslayer_input (torch.tensor): A tensor representing the input features of the previous layer.
            fowardlayerrelevance (torch.tensor): A tensor representing the relevance of the features in the forward layer.
            rule (str, optional): A string specifying the LRP rule to be used. Default is "lrp0".
            parameters (dict, optional): A dictionary containing additional parameters for specific LRP rules.

        Returns:
            torch.tensor: A tensor representing the relevance of the previous layer's input features.
        """
        Aij=previouslayer_input.data.requires_grad_(True)
        if rule=="lrp0":
            Zk=super(Conv2DLRP, self).forward(Aij)     #equation 7.1 from chapter 7 (zk=ε + ∑0,j aj · wjk )
            sensitivity=(fowardlayerrelevance/Zk).data #equation 7.2 from chapter 7 (sk=Rk/zk)
            (Zk*sensitivity).sum().backward()          # calculating the gradients (backpropagating the forward layer value to present layer inputs)
            layerrelevance=(Aij*Aij.grad).data         # extracting the relevance values
            return layerrelevance
        elif rule=="lrpepsilon":
            Zk=super(Conv2DLRP, self).forward(Aij) #equation 7.1 from chapter 7 (zk=ε + ∑0,j aj · wjk )
            Zk += torch.sign(Zk)*parameters["epsilon"]  # implementing epsilon rule 
            sensitivity=(fowardlayerrelevance/Zk).data #equation 7.2 from chapter 7 (sk=Rk/zk)
            (Zk*sensitivity).sum().backward()  # calculating the gradients (backpropagating the forward layer value to present layer inputs)
            layerrelevance=(Aij*Aij.grad).data # extracting the relevance values
            return layerrelevance
        elif rule=="lrpzplus": #implemented the rule based on the Table 3.1 in chapeter 3
            self.weight = torch.nn.Parameter(self.weight.clamp(min=0.0))
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))
            Zk=super(Conv2DLRP, self).forward(Aij)+parameters["epsilon"] 
            Zk.clamp(min=0)
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        elif rule=="lrpgamma":
            gweight = self.weight+ self.weight.clamp(min=0)* parameters["gamma"]
            gbias = self.bias+self.bias.clamp(min=0) * parameters["gamma"]
            Zk= F.conv2d(Aij, gweight, gbias, self.stride, self.padding, self.dilation, self.groups) 
            Zk += torch.sign(Zk)*parameters["epsilon"] 
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        elif rule=="lrpalphabeta":  #implemented the rule based on the Table 3.1 in chapeter 3
            output_shape=Aij.shape
            relevance_output_shape = fowardlayerrelevance.shape
            weight_shape = self.weight.shape
            relevance_input_shape = (
                relevance_output_shape[0],                              
                weight_shape[1],                                       
                (relevance_output_shape[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (weight_shape[2] - 1) + 1,
                (relevance_output_shape[3] - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (weight_shape[3] - 1) + 1
            )#calculating the key infor to perform back propagation of convolution layer using convolution transpose 
            if relevance_input_shape!=output_shape: 
                output_padding = (output_shape[-2]-relevance_input_shape[-2],output_shape[-1]-relevance_input_shape[-1])
            else:
                output_padding =0# the weight, activation are seperated based on the sign and factor is multipled based on alpha and beta value
            pAij=Aij.clamp(min=0)
            nAij=Aij.clamp(max=0)
            pweights = self.weight.clamp(min=0)
            nweights = self.weight.clamp(max=0)
            if parameters["gamma"]!=0:
                pweights +=pweights.clamp(min=0)* parameters["gamma"]
                nweights +=nweights.clamp(max=0)* parameters["gamma"]
            pZ1=F.conv2d(pAij, pweights, None, self.stride, self.padding, self.dilation, self.groups).clamp(min=0)
            nZ1=F.conv2d(nAij,nweights, None, self.stride, self.padding, self.dilation, self.groups).clamp(max=0)
            pZ1 += torch.sign(pZ1)*parameters["epsilon"] 
            nZ1 += torch.sign(nZ1)*parameters["epsilon"] 
            Rp = parameters["alpha"]  *(fowardlayerrelevance / (pZ1+(pZ1 == 0).float()))
            Rn = parameters["beta"] *(fowardlayerrelevance / (nZ1+(nZ1 == 0).float()))
            pRi=F.conv_transpose2d(Rp, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
            nRi=F.conv_transpose2d(Rn, nweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
            layerrelevance = ( (pRi*pAij) )- ((nRi*nAij))
            return layerrelevance.data
        elif rule=="lrpzbetalh":  #implemented the rule based on the Table 3.1 in chapeter 3
            lb = (Aij.detach()*0-1).requires_grad_(True)
            hb = (Aij.detach()*0+1).requires_grad_(True)
            pweights=self.weight.clamp(min=0)
            nweights=self.weight.clamp(max=0)
            Zk=super(Conv2DLRP, self).forward(Aij)
            Zk +=parameters["epsilon"]
            lZ=F.conv2d(lb, pweights, None, self.stride, self.padding, self.dilation, self.groups)
            hZ=F.conv2d(hb, nweights, None, self.stride, self.padding, self.dilation, self.groups)
            Zk=Zk-lZ-hZ
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad+lb*lb.grad+hb*hb.grad)
            return  layerrelevance
        else: #init type based on the pseudo code from Explainable AI: Interpreting, Explaining and Visualizing Deep Learning Book
            Zk=super(Conv2DLRP, self).forward(Aij)
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance