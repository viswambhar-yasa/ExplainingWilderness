import torch 
import torch.nn as nn

class MaxPool2dLRP(nn.MaxPool2d):
    def __init__(self, maxpool_layer: torch.nn.MaxPool2d) -> None:
        super().__init__(maxpool_layer.kernel_size, maxpool_layer.stride, maxpool_layer.padding, maxpool_layer.dilation, maxpool_layer.return_indices, maxpool_layer.ceil_mode)
        self.copy_parameters(maxpool_layer)

    def copy_parameters(self, module):
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
    

    def forward(self,input:torch.tensor):
        return super(MaxPool2dLRP, self).forward(input)
    
    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)  
        if rule=="lrpzplus" or rule=="lrpepsilon":
            Zk=super(MaxPool2dLRP, self).forward(Aij)
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
            Zk=super(MaxPool2dLRP, self).forward(Aij)
            fowardlayerrelevance=fowardlayerrelevance.view(Zk.shape)
            sensitivity=(fowardlayerrelevance/Zk).data
            sensitivity[Zk==0]=0
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance 