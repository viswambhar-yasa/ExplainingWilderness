import torch 
import torch.nn as nn
import torch.nn.functional as F
from .ConceptMask import RelevanceConcepts
class LinearCRP(nn.Linear):
    def __init__(self, linear_layer: nn.Linear) -> None:
        super().__init__(linear_layer.in_features, linear_layer.out_features)
        self.copy_parameters(linear_layer)
        self.conceptmask=RelevanceConcepts()
        self.conceptmask.register_channels(int(linear_layer.out_features))
        self.concept_index=None

    def copy_parameters(self, module):
        with torch.no_grad():
            self.weight.data.copy_(module.weight.data)
            if self.bias is not None and module.bias is not None:
                self.bias.data.copy_(module.bias.data)

    def forward(self,input:torch.tensor):
        return super(LinearCRP, self).forward(input)
    

    def interpet(self,previouslayer_input: torch.tensor, fowardlayerrelevance:dict,concepts=None,conceptindex_estimation=None,top_num=2,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)   
        outputconceptrelevance={}
        conceptestimated=False
        if conceptindex_estimation=="relmax":
            self.concept_index=self.conceptmask.max_relevance_concept(fowardlayerrelevance,top_num)
            conceptestimated=True
        if rule=="lrpzplus":
            pAij=Aij.clamp(min=0)
            pweights = self.weight.clamp(min=0)
            if parameters["gamma"]!=0:
                pweights +=pweights.clamp(min=0)*parameters["gamma"]
            pZ = F.linear(pAij, pweights, bias=None).clamp(min=0)
            pZ += torch.sign(pZ)*parameters["epsilon"] 
            if concepts is not None:
                self.conceptmask.register_concept_channel(concepts,pZ.shape)
                for concept_index,conceptmask in self.conceptmask.mask.items():
                    for index,conceptrelevance in fowardlayerrelevance.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        #print((conceptrelevance*conceptmask).sum())
                        sensitivity=((conceptrelevance*conceptmask) / (pZ+(pZ == 0).float()))
                        pR = F.linear(sensitivity, pweights.t(), bias=None)
                        layerrelevance=pR*pAij
                        outputconceptrelevance[concept_name]=layerrelevance
                self.conceptmask.deregister_concept_channel()
            if conceptestimated:
                for index,conceptrelevance in fowardlayerrelevance.items():
                    estimated_indicies=self.concept_index[index].keys()
                    print(estimated_indicies)
                    self.conceptmask.register_batch_concept_channel(estimated_indicies,pZ.shape)
                    for concept_index,conceptmask in self.conceptmask.mask.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        #print((conceptrelevance*conceptmask).sum())
                        sensitivity=((conceptrelevance*conceptmask) / (pZ+(pZ == 0).float()))
                        pR = F.linear(sensitivity, pweights.t(), bias=None)
                        layerrelevance=pR*pAij
                        outputconceptrelevance[concept_name]=layerrelevance
                    self.conceptmask.deregister_concept_channel()
                return outputconceptrelevance   
            else:
                for index,conceptrelevance in fowardlayerrelevance.items():
                    sensitivity=(conceptrelevance / (pZ+(pZ == 0).float()))
                    pR = F.linear(sensitivity, pweights.t(), bias=None)
                    layerrelevance=pR*pAij
                    outputconceptrelevance[index]=layerrelevance
                return outputconceptrelevance
            
        elif rule=="lrpepsilon":
            Zk=super(LinearCRP, self).forward(Aij)
            Zk += torch.sign(Zk)*parameters["epsilon"] 
            if concepts is not None:
                self.conceptmask.register_concept_channel(concepts,Zk.shape)
                for concept_index,conceptmask in self.conceptmask.mask.items():
                    for index,conceptrelevance in fowardlayerrelevance.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        sensitivity=((conceptrelevance*conceptmask) / (Zk+(Zk == 0).float()))
                        sensitivity[Zk==0]=0
                        R = F.linear(sensitivity, self.weight.t(), bias=None)
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
                        R = F.linear(sensitivity, self.weight.t(), bias=None)
                        layerrelevance=R*Aij
                        outputconceptrelevance[concept_name]=layerrelevance
                    self.conceptmask.deregister_concept_channel()
                return outputconceptrelevance
            else:
                for index,conceptrelevance in fowardlayerrelevance.items():
                        sensitivity=(conceptrelevance / (Zk+(Zk == 0).float()))
                        sensitivity[Zk==0]=0
                        R = F.linear(sensitivity, self.weight.t(), bias=None)
                        layerrelevance=R*Aij
                        outputconceptrelevance[index]=layerrelevance
                return outputconceptrelevance
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
            if concepts is not None:
                self.conceptmask.register_concept_channel(concepts,pZ1.shape)
                for concept_index,conceptmask in self.conceptmask.mask.items():
                    for index,conceptrelevance in fowardlayerrelevance.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        Rp = parameters["alpha"]  *((conceptrelevance*conceptmask) / (pZ1+(pZ1 == 0).float()))
                        Rn = parameters["beta"]  * ((conceptrelevance*conceptmask) / (nZ1+(nZ1 == 0).float()))
                        pRi2 = F.linear(Rp, pweights.t(), bias=None)
                        nRi2 = F.linear(Rn, nweights.t(), bias=None)
                        layerrelevance = ( (pRi2*pAij)) - ((nRi2*nAij))
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
                        pRi2 = F.linear(Rp, pweights.t(), bias=None)
                        nRi2 = F.linear(Rn, nweights.t(), bias=None)
                        layerrelevance = ( (pRi2*pAij)) - ((nRi2*nAij))
                        #layerrelevance /= (layerrelevance.sum() + 1e-7)
                        outputconceptrelevance[concept_name]=layerrelevance.data
                    self.conceptmask.deregister_concept_channel()
                return outputconceptrelevance
            else:
                for index,conceptrelevance in fowardlayerrelevance.items():
                        Rp = parameters["alpha"]  *(conceptrelevance / (pZ1+(pZ1 == 0).float()))
                        Rn = parameters["beta"]  * (conceptrelevance / (nZ1+(nZ1 == 0).float()))
                        pRi2 = F.linear(Rp, pweights.t(), bias=None)
                        nRi2 = F.linear(Rn, nweights.t(), bias=None)
                        layerrelevance = ( (pRi2*pAij)) - ((nRi2*nAij))
                        #layerrelevance /= (layerrelevance.sum() + 1e-7)
                        outputconceptrelevance[index]=layerrelevance 
                return outputconceptrelevance
