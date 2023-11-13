import torch 
import torch.nn as nn
import torch.nn.functional as F
class LinearLRP(nn.Linear):
    def __init__(self, linear_layer: nn.Linear) -> None:
        super().__init__(linear_layer.in_features, linear_layer.out_features)
        self.copy_parameters(linear_layer)

    def copy_parameters(self, module):
        with torch.no_grad():
            self.weight.data.copy_(module.weight.data)
            if self.bias is not None and module.bias is not None:
                self.bias.data.copy_(module.bias.data)

    def forward(self,input:torch.tensor):
        return super(LinearLRP, self).forward(input)
    
    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)   
        
        if rule=="lrpepsilon":
            Zk=super(LinearLRP, self).forward(Aij)
            Zk += torch.sign(Zk)*parameters["epsilon"] 
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        elif rule=="lrpzplus":
            self.weight = torch.nn.Parameter(self.weight.clamp(min=0.0))
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))
            
            Zk=super(LinearLRP, self).forward(Aij)
            Zk.clamp(min=0)
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        elif rule=="lrpgamma":
            gweight = self.weight+ self.weight.clamp(min=0)* parameters["gamma"]
            gbias = self.bias+self.bias.clamp(min=0) * parameters["gamma"]
            Zk= F.linear(Aij, gweight, gbias)
            Zk += torch.sign(Zk)*parameters["epsilon"] 
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        elif rule=="lrpalphabeta":
            pAij=Aij.clamp(min=0)
            nAij=Aij.clamp(max=0)
            pweights = self.weight.clamp(min=0)
            nweights = self.weight.clamp(max=0)
            if parameters["gamma"]!=0:
                pweights +=pweights.clamp(min=0)*parameters["gamma"]
                nweights +=nweights.clamp(max=0)* parameters["gamma"]
            pZ1 = F.linear(pAij, pweights, bias=None).clamp(min=0)
            nZ1 = F.linear(nAij, nweights, bias=None).clamp(max=0)
            pZ1 += torch.sign(pZ1)*parameters["epsilon"] 
            nZ1 += torch.sign(nZ1)*parameters["epsilon"] 
            Rp = parameters["alpha"]  *(fowardlayerrelevance / (pZ1+(pZ1 == 0).float()))
            Rn = parameters["beta"]  * (fowardlayerrelevance / (nZ1+(nZ1 == 0).float()))
            pRi2 = F.linear(Rp, pweights.t(), bias=None)
            nRi2 = F.linear(Rn, nweights.t(), bias=None)
            layerrelevance = ( (pRi2*pAij)) - ((nRi2*nAij))
            return layerrelevance.data
        else:
            Zk=super(LinearLRP, self).forward(Aij)
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        
        