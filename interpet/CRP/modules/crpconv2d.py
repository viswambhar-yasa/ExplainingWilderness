import torch 
import torch.nn as nn
import torch.nn.functional as F
from .ConceptMask import RelevanceConcepts
class Conv2DCRP(nn.Conv2d):
    def __init__(self, conv_layer: nn.Conv2d):
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, conv_layer.dilation, conv_layer.groups, conv_layer.bias is not None, conv_layer.padding_mode)
        self.copy_parameters(conv_layer)
        #self.conceptmask=RelevanceConcepts(out_channels=int(conv_layer.out_channels))
        self.conceptmask=RelevanceConcepts()
        self.conceptmask.register_channels(int(conv_layer.out_channels))
        self.concept_index=None

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
        return super(Conv2DCRP, self).forward(input)
    
    def interpet(self,previouslayer_input: torch.tensor, fowardlayerrelevance:dict,concepts=None,conceptindex_estimation=None,top_num=2,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)   
        outputconceptrelevance={}
        conceptestimated=False
        if conceptindex_estimation=="relmax":
            self.concept_index=self.conceptmask.max_relevance_concept(fowardlayerrelevance,top_num)
            conceptestimated=True
        
        output_shape=Aij.shape
        first_key = next(iter(fowardlayerrelevance), None)  # Get the first key
        first_value = fowardlayerrelevance.get(first_key, None)
        relevance_output_shape = first_value.shape
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
        if rule=="lrpzplus":
            pAij=Aij.clamp(min=0)
            pweights = self.weight.clamp(min=0)
            if parameters["gamma"]!=0:
                pweights +=pweights.clamp(min=0)*parameters["gamma"]
            pZ=F.conv2d(pAij, pweights, None, self.stride, self.padding, self.dilation, self.groups).clamp(min=0)
            pZ += torch.sign(pZ)*parameters["epsilon"] 
            if concepts is not None:
                self.conceptmask.register_concept_channel(concepts,pZ.shape)
                for concept_index,conceptmask in self.conceptmask.mask.items():
                    for index,conceptrelevance in fowardlayerrelevance.items():
                        #print((conceptrelevance*conceptmask).sum())
                        concept_name=str(index)+"_"+str(concept_index)
                        sensitivity=((conceptrelevance*conceptmask) / (pZ+(pZ == 0).float()))
                        pR=F.conv_transpose2d(sensitivity, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance=pR*pAij
                        outputconceptrelevance[concept_name]=layerrelevance
                self.conceptmask.deregister_concept_channel()
            if conceptestimated:
                for index,conceptrelevance in fowardlayerrelevance.items():
                    estimated_indicies=self.concept_index[index].keys()
                    self.conceptmask.register_batch_concept_channel(estimated_indicies,pZ.shape)
                    for concept_index,conceptmask in self.conceptmask.mask.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        sensitivity=((conceptrelevance*conceptmask) / (pZ+(pZ == 0).float()))
                        pR=F.conv_transpose2d(sensitivity, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance=pR*pAij
                        outputconceptrelevance[concept_name]=layerrelevance
                    self.conceptmask.deregister_concept_channel()
                return outputconceptrelevance
            else:
                for index,conceptrelevance in fowardlayerrelevance.items():
                        #print((conceptrelevance).sum())
                        sensitivity=(conceptrelevance / (pZ+(pZ == 0).float()))
                        pR=F.conv_transpose2d(sensitivity, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance=pR*pAij
                        outputconceptrelevance[index]=layerrelevance
                return outputconceptrelevance
        elif rule=="lrpalphabeta":
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
            if concepts is not None:
                self.conceptmask.register_concept_channel(concepts,pZ1.shape)
                for concept_index,conceptmask in self.conceptmask.mask.items():
                    for index,conceptrelevance in fowardlayerrelevance.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        Rp = parameters["alpha"]  *((conceptrelevance*conceptmask) / (pZ1+(pZ1 == 0).float()))
                        Rn = parameters["beta"]  * ((conceptrelevance*conceptmask) / (nZ1+(nZ1 == 0).float()))
                        pRi=F.conv_transpose2d(Rp, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        nRi=F.conv_transpose2d(Rn, nweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance = ( (pRi*pAij) )- ((nRi*nAij))
                        #layerrelevance /= (layerrelevance.sum() + 1e-7)
                        outputconceptrelevance[concept_name]=layerrelevance.data
                self.conceptmask.deregister_concept_channel()
            if conceptestimated:
                for index,conceptrelevance in fowardlayerrelevance.items():
                    estimated_indicies=self.concept_index[index].keys()
                    self.conceptmask.register_batch_concept_channel(estimated_indicies,pZ1.shape)
                    for concept_index,conceptmask in self.conceptmask.mask.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        Rp = parameters["alpha"]  *((conceptrelevance*conceptmask) / (pZ1+(pZ1 == 0).float()))
                        Rn = parameters["beta"]  * ((conceptrelevance*conceptmask) / (nZ1+(nZ1 == 0).float()))
                        pRi=F.conv_transpose2d(Rp, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        nRi=F.conv_transpose2d(Rn, nweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance = ( (pRi*pAij) )- ((nRi*nAij))
                        #layerrelevance /= (layerrelevance.sum() + 1e-7)
                        outputconceptrelevance[concept_name]=layerrelevance.data
                    self.conceptmask.deregister_concept_channel()
                return outputconceptrelevance
            else:
                for index,conceptrelevance in fowardlayerrelevance.items():
                        Rp = parameters["alpha"]  *(conceptrelevance / (pZ1+(pZ1 == 0).float()))
                        Rn = parameters["beta"]  * (conceptrelevance / (nZ1+(nZ1 == 0).float()))
                        pRi=F.conv_transpose2d(Rp, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        nRi=F.conv_transpose2d(Rn, nweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance = ( (pRi*pAij) )- ((nRi*nAij))
                        #layerrelevance /= (layerrelevance.sum() + 1e-7)
                        outputconceptrelevance[index]=layerrelevance
                return outputconceptrelevance
            
        else:
            Zk=super(Conv2DCRP, self).forward(Aij)
            Zk += torch.sign(Zk)*parameters["epsilon"] 
            if concepts is not None:
                self.conceptmask.register_concept_channel(concepts,Zk.shape)
                for concept_index,conceptmask in self.conceptmask.mask.items():
                    for index,conceptrelevance in fowardlayerrelevance.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        sensitivity=((conceptrelevance*conceptmask) / (Zk+(Zk == 0).float()))
                        sensitivity[Zk==0]=0
                        R=F.conv_transpose2d(sensitivity, self.weight, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance=R*Aij
                        outputconceptrelevance[concept_name]=layerrelevance
                self.conceptmask.deregister_concept_channel()
            if conceptestimated:
                for index,conceptrelevance in fowardlayerrelevance.items():
                    estimated_indicies=self.concept_index[index].keys()
                    self.conceptmask.register_batch_concept_channel(estimated_indicies,Zk.shape)
                    for concept_index,conceptmask in self.conceptmask.mask.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        sensitivity=((conceptrelevance*conceptmask) / (Zk+(Zk == 0).float()))
                        sensitivity[Zk==0]=0
                        R=F.conv_transpose2d(sensitivity, self.weight, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance=R*Aij
                        outputconceptrelevance[concept_name]=layerrelevance
                    self.conceptmask.deregister_concept_channel()
                return outputconceptrelevance  
            else:
                for index,conceptrelevance in fowardlayerrelevance.items():
                        sensitivity=(conceptrelevance / (Zk+(Zk == 0).float()))
                        sensitivity[Zk==0]=0
                        R=F.conv_transpose2d(sensitivity, self.weight, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance=R*Aij
                        outputconceptrelevance[index]=layerrelevance
                return outputconceptrelevance
