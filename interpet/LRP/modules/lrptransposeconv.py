import torch 
import torch.nn as nn
import torch.nn.functional as F


class ConvTranspose2DLRP(nn.ConvTranspose2d):
    def __init__(self, convtranspose_layer: nn.ConvTranspose2d):
        super().__init__(convtranspose_layer.in_channels,convtranspose_layer.out_channels,convtranspose_layer.kernel_size,
                         convtranspose_layer.stride,convtranspose_layer.padding,convtranspose_layer.output_padding,
                         convtranspose_layer.groups,convtranspose_layer.dilation)
        self.copy_parameters(convtranspose_layer)

    def copy_parameters(self, module):
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
        return super(ConvTranspose2DLRP, self).forward(input)

    def interpet(self, previouslayer_input: torch.tensor, fowardlayerrelevance: torch.tensor,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)
        Zk=super(ConvTranspose2DLRP, self).forward(Aij)
        sensitivity=(fowardlayerrelevance/Zk).data
        sensitivity[Zk==0]=0
        (Zk*sensitivity).sum().backward()
        layerrelevance=(Aij*Aij.grad).data
        return layerrelevance
    