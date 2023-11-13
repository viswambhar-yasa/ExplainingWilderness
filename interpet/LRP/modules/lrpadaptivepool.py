import torch 
import torch.nn as nn
import torch.nn.functional as F
class AdaptiveAvgPool2dLRP(nn.AdaptiveAvgPool2d):
    def __init__(self, avgpool_layer: nn.AdaptiveAvgPool2d) -> None:
        super().__init__(avgpool_layer.output_size)
        self.output_size = avgpool_layer.output_size
    
    def forward(self,input:torch.tensor):
        return super(AdaptiveAvgPool2dLRP, self).forward(input)
    
    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)
        if rule=="lrpzplus" or rule=="lrpepsilon":
            Zk=super(AdaptiveAvgPool2dLRP, self).forward(Aij)
            fowardlayerrelevance=fowardlayerrelevance.view(Zk.shape)
            if rule=="lrpzplus":
                Zk=Zk.clamp(min=0)
            Zk += torch.sign(Zk)*parameters["epsilon"]
            sensitivity=(fowardlayerrelevance/Zk).data
            sensitivity[Zk==0]=0
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance 
        else:
            Zk=super(AdaptiveAvgPool2dLRP, self).forward(Aij)
            fowardlayerrelevance=fowardlayerrelevance.view(Zk.shape)
            sensitivity=(fowardlayerrelevance/Zk).data
            sensitivity[Zk==0]=0
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance 
        