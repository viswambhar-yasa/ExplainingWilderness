import torch 
import torch.nn as nn
import torch.nn.functional as F
class AdaptiveAvgPool2dCRP(nn.AdaptiveAvgPool2d):
    def __init__(self, avgpool_layer: nn.AdaptiveAvgPool2d) -> None:
        super().__init__(avgpool_layer.output_size)
        self.output_size = avgpool_layer.output_size
    
    def forward(self,input:torch.tensor):
        return super(AdaptiveAvgPool2dCRP, self).forward(input)
    
    def interpet(self,previouslayer_input: torch.tensor, fowardlayerrelevance:dict,concepts=None,conceptindex_estimation=None,top_num=2,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)
        outputconceptrelevance={}
        Zk=super(AdaptiveAvgPool2dCRP, self).forward(Aij)
        n,c,h,w=Aij.shape
        n,C,H,W=Zk.shape
        stride=(int(h/H),int(w/W))
        kernel_size=(h-(H-1)*stride[0],(w-(W-1)*stride[1]))
        weight=torch.ones(size=(c,C,kernel_size[0],kernel_size[1]))/(kernel_size[0]*kernel_size[1])
        if rule=="lrpzplus" or rule=="lrpepsilon":
            if rule=="lrpzplus":
                Zk=Zk.clamp(min=0)
            Zk += torch.sign(Zk)*parameters["epsilon"]
        for index,conceptrelevance in fowardlayerrelevance.items():
            #print(conceptrelevance.sum())
            conceptrelevance=conceptrelevance.view(Zk.shape)
            sensitivity=(conceptrelevance/Zk)
            sensitivity[Zk==0]=0
            pR=F.conv_transpose2d(sensitivity, weight, None,stride=stride,padding=0)
            layerrelevance=pR*Aij
            layerrelevance /= (layerrelevance.sum() + 1e-7)
            outputconceptrelevance[index]=layerrelevance
        return outputconceptrelevance
            
        