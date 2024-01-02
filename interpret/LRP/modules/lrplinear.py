# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented layer wise propagation method for linear layer, the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: Built based on the base lrp module present in lrp from Explainable AI: Interpreting, Explaining and Visualizing Deep Learning Book

import torch 
import torch.nn as nn
import torch.nn.functional as F


class LinearLRP(nn.Linear):
    """
    LinearLRP class is a subclass of nn.Linear in PyTorch. It is used for interpreting the relevance of input features in a linear layer of a neural network using different LRP (Layer-wise Relevance Propagation) rules.
    """

    def __init__(self, linear_layer: nn.Linear) -> None:
        """
        Initializes the LinearLRP object by copying the parameters from the given linear layer.

        Args:
            linear_layer (nn.Linear): The linear layer from which to copy the parameters.
        """
        super().__init__(linear_layer.in_features, linear_layer.out_features)
        self.copy_parameters(linear_layer)

    def copy_parameters(self, module):
        """
        Copies the parameters (weights and biases) from a given module to the LinearLRP object.

        Args:
            module: The module from which to copy the parameters.
        """
        with torch.no_grad():
            self.weight.data.copy_(module.weight.data)
            if self.bias is not None and module.bias is not None:
                self.bias.data.copy_(module.bias.data)

    def forward(self, input: torch.tensor):
        """
        Performs forward propagation using the inherited forward method.

        Args:
            input (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor.
        """
        return super(LinearLRP, self).forward(input)
    
    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor, rule="lrp0", parameters={}) -> torch.tensor:
        """
        Interprets the relevance of input features based on the chosen LRP rule and returns the relevance tensor.

        Args:
            previouslayer_input (torch.tensor): The input tensor to the previous layer.
            fowardlayerrelevance (torch.tensor): The relevance tensor of the forward layer.
            rule (str, optional): The LRP rule to use. Defaults to "lrp0".
            parameters (dict, optional): Additional parameters for the LRP rule. Defaults to {}.

        Returns:
            torch.tensor: The relevance tensor.
        """
        Aij = previouslayer_input.data.requires_grad_(True)   
       



        if rule == "lrpepsilon":
            Zk = super(LinearLRP, self).forward(Aij)       #equation 7.1 from chapter 7 (zk=ε + ∑0,j aj · wjk )      
            Zk += torch.sign(Zk) * parameters["epsilon"]   #equation 7.1 from chapter 7 adding epsilon
            sensitivity = (fowardlayerrelevance / Zk).data #equation 7.2 from chapter 7 (sk=Rk/zk)
            (Zk * sensitivity).sum().backward()            # calculating the gradients (backpropagating the forward layer to present layer inputs)
            layerrelevance = (Aij * Aij.grad).data         # extracting the relevance values R_j = a_j ·  c_j
            return layerrelevance
        elif rule == "lrpzplus": #implemented the rule based on the Table 3.1 in chapeter 3
            self.weight = torch.nn.Parameter(self.weight.clamp(min=0.0))
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))
            #only the poisitive weights and activation are selected 
            Zk = super(LinearLRP, self).forward(Aij)
            Zk.clamp(min=0)
            sensitivity = (fowardlayerrelevance / Zk).data
            (Zk * sensitivity).sum().backward()
            layerrelevance = (Aij * Aij.grad).data
            return layerrelevance
        elif rule == "lrpgamma": #apply power law to weight, to show case their influence on the prediction 
            gweight = self.weight + self.weight.clamp(min=0) * parameters["gamma"]
            gbias = self.bias + self.bias.clamp(min=0) * parameters["gamma"]
            Zk = F.linear(Aij, gweight, gbias)
            Zk += torch.sign(Zk) * parameters["epsilon"] 
            sensitivity = (fowardlayerrelevance / Zk).data
            (Zk * sensitivity).sum().backward()
            layerrelevance = (Aij * Aij.grad).data
            return layerrelevance
        elif rule == "lrpalphabeta":   #implemented the rule based on the Table 3.1 in chapeter 3
            # the weight, activation are seperated based on the sign and factor is multipled based on alpha and beta value and step 7.1,7.2,7.3 and 7.4 are performed
            pAij = Aij.clamp(min=0)
            nAij = Aij.clamp(max=0)
            pweights = self.weight.clamp(min=0)
            nweights = self.weight.clamp(max=0)
            if parameters["gamma"] != 0:
                pweights += pweights.clamp(min=0) * parameters["gamma"]
                nweights += nweights.clamp(max=0) * parameters["gamma"]
            
            pZ1 = F.linear(pAij, pweights, bias=None).clamp(min=0)
            nZ1 = F.linear(nAij, nweights, bias=None).clamp(max=0)
            pZ1 += torch.sign(pZ1) * parameters["epsilon"] 
            nZ1 += torch.sign(nZ1) * parameters["epsilon"] 
            Rp = parameters["alpha"] * (fowardlayerrelevance / (pZ1 + (pZ1 == 0).float()))
            Rn = parameters["beta"] * (fowardlayerrelevance / (nZ1 + (nZ1 == 0).float()))
            pRi2 = F.linear(Rp, pweights.t(), bias=None)
            nRi2 = F.linear(Rn, nweights.t(), bias=None)
            layerrelevance = ( (pRi2*pAij)) - ((nRi2*nAij))
            return layerrelevance.data
        else:#default mode init propagation
            Zk=super(LinearLRP, self).forward(Aij)
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        
        