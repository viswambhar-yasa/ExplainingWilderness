import torch 
import torch.nn as nn
import torch.nn.functional as F
class Conv2DLRP(nn.Conv2d):
    def __init__(self, conv_layer: nn.Conv2d):
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, conv_layer.dilation, conv_layer.groups, conv_layer.bias is not None, conv_layer.padding_mode)
        self.copy_parameters(conv_layer)

    def copy_parameters(self, module):
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
    
    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)
        if rule=="lrp0":
            Zk=super(Conv2DLRP, self).forward(Aij)
            sensitivity=(fowardlayerrelevance/Zk).data
            #sensitivity[Zk==0]=0
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        elif rule=="lrpepsilon":
            Zk=super(Conv2DLRP, self).forward(Aij)
            Zk += torch.sign(Zk)*parameters["epsilon"] 
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance
        elif rule=="lrpzplus":
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
        elif rule=="lrpalphabeta":
            output_shape=Aij.shape
            relevance_output_shape = fowardlayerrelevance.shape
            weight_shape = self.weight.shape
            relevance_input_shape = (
                relevance_output_shape[0],                              
                weight_shape[1],                                       
                (relevance_output_shape[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (weight_shape[2] - 1) + 1,
                (relevance_output_shape[3] - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (weight_shape[3] - 1) + 1
            )
            if relevance_input_shape!=output_shape:
                output_padding = (output_shape[-2]-relevance_input_shape[-2],output_shape[-1]-relevance_input_shape[-1])
            else:
                output_padding =0
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
        elif rule=="lrpzbetalh":
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
        else:
            Zk=super(Conv2DLRP, self).forward(Aij)
            sensitivity=(fowardlayerrelevance/Zk).data
            (Zk*sensitivity).sum().backward()
            layerrelevance=(Aij*Aij.grad).data
            return layerrelevance