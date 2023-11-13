import torch 
import torch.nn as nn

class MaxPool2dCRP(nn.MaxPool2d):
    def __init__(self, maxpool_layer: torch.nn.MaxPool2d) -> None:
        super().__init__(maxpool_layer.kernel_size, maxpool_layer.stride, maxpool_layer.padding, maxpool_layer.dilation, maxpool_layer.return_indices, maxpool_layer.ceil_mode)
        self.copy_parameters(maxpool_layer)
        self.return_indices=True

    def copy_parameters(self, module):
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
    
    def forward(self,input:torch.tensor):
        x,self.indicies=super(MaxPool2dCRP, self).forward(input)
        return x
    
    def interpet(self,previouslayer_input: torch.tensor, fowardlayerrelevance:dict,concepts=None,conceptindex_estimation=None,top_num=2,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True) 
        outputconceptrelevance={} 
        Zk,_=super(MaxPool2dCRP, self).forward(Aij)
        outputsize=Aij.shape
        
        if rule=="lrpzplus" or rule=="lrpepsilon":
            Zk=Zk.clamp(min=0)
            Zk += torch.sign(Zk)*parameters["epsilon"] 
        
        for index,conceptrelevance in fowardlayerrelevance.items():
            #print(conceptrelevance.sum())
            conceptrelevance=conceptrelevance.view(Zk.shape)
            sensitivity=(conceptrelevance/Zk)
            sensitivity[Zk==0]=0
            R=torch.nn.functional.max_unpool2d(sensitivity,self.indicies,self.kernel_size,self.stride,self.padding,outputsize)
            layerrelevance=(R*Aij)
            #layerrelevance /= (layerrelevance.sum() + 1e-7)
            outputconceptrelevance[index]=layerrelevance
                
        return outputconceptrelevance
        