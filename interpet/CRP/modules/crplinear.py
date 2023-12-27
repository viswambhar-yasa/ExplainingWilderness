# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented concept relevance propagation method for linear layer, the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from interpet.CRP.modules.ConceptMask import RelevanceConcepts


class LinearCRP(nn.Linear):
    
    """
    The `LinearCRP` class is a subclass of `nn.Linear` that adds additional functionality for interpreting the output of the linear layer. It includes methods for interpreting the relevance of input features and concepts, based on different interpretation rules.

    Example Usage:
        linear_layer = nn.Linear(10, 5)
        crp = LinearCRP(linear_layer)
        input = torch.randn(1, 10)
        output = crp(input)
        relevance = {"0": torch.randn(1, 5), "1": torch.randn(1, 5)}
        concept_relevance = crp.interpet(input, relevance, concepts=[0, 1], top_num=2, rule="lrp0")

    Methods:
        - __init__(self, linear_layer: nn.Linear) -> None: Initializes the `LinearCRP` object by copying the parameters of the given linear layer and registering concept channels.
        - copy_parameters(self, module): Copies the parameters (weights and biases) from a given module to the `LinearCRP` object.
        - forward(self, input: torch.tensor) -> torch.tensor: Performs the forward pass of the linear layer.
        - interpret(self, previouslayer_input: torch.tensor, forwardlayer_relevance: dict, concepts=None, conceptindex_estimation=None, top_num=2, rule="lrp0", parameters={}) -> torch.tensor: Interprets the relevance of input features and concepts based on the specified interpretation rule. It returns a dictionary of output concept relevance.

    Fields:
        - conceptmask: An instance of the `RelevanceConcepts` class for managing concept channels and calculating maximum relevance concepts.
        - concept_index: A dictionary that stores the indices of the maximum relevance concepts for each channel.
    """
    def __init__(self, linear_layer: nn.Linear) -> None:
        super().__init__(linear_layer.in_features, linear_layer.out_features)
        self.copy_parameters(linear_layer)
        self.conceptmask=RelevanceConcepts()
        self.conceptmask.register_channels(int(linear_layer.out_features))
        self.concept_index=None

    def copy_parameters(self, module):
        """
        Copies the parameters (weights and biases) from a given module to the LinearCRP object.

        Args:
            module (nn.Module): The module from which the parameters will be copied.

        Returns:
            None

        """
        with torch.no_grad():
            self.weight.data.copy_(module.weight.data)
            if self.bias is not None and module.bias is not None:
                self.bias.data.copy_(module.bias.data)

    def forward(self, input: torch.tensor):
        """
        Overrides the forward method of the parent class nn.Linear and returns the output of the linear layer.

        Args:
            input (torch.tensor): A tensor representing the input to the linear layer.

        Returns:
            torch.tensor: The output of the linear layer.
        """
        return super(LinearCRP, self).forward(input)

    """
    Interprets the relevance of input features and concepts based on different interpretation rules.

    Inputs:
    - previouslayer_input (torch.tensor): The input tensor from the previous layer.
    - forwardlayer_relevance (dict): A dictionary containing the relevance values for each output concept.
    - concepts (list): A list of concept indices to consider for interpretation (optional).
    - conceptindex_estimation (str): The method used to estimate the concept indices (optional).
    - top_num (int): The number of top concepts to consider for estimation (optional).
    - rule (str): The interpretation rule to use (optional).
    - parameters (dict): Additional parameters for the interpretation rule (optional).

    Returns:
    - outputconceptrelevance (dict): A dictionary containing the interpreted relevance values for each output concept.
    """
    def interpet(self,previouslayer_input: torch.tensor, fowardlayerrelevance:dict,concepts=None,conceptindex_estimation=None,top_num=2,rule="lrp0",parameters={}) -> torch.tensor:
        Aij=previouslayer_input.data.requires_grad_(True)   
        outputconceptrelevance={} # dictionary to store relevance map, as each new concept need to be have it own path, we seperate them and store them in 
        conceptestimated=False
        if conceptindex_estimation=="relmax": #if relevance maximum needs to be calculated for the forward layer, where the max relevance channels are selected
            self.concept_index=self.conceptmask.max_relevance_concept(fowardlayerrelevance,top_num)
            conceptestimated=True
        if rule=="lrpzplus": #implemented the rule based on the Table 3.1 in chapeter 3
            pAij=Aij.clamp(min=0)
            pweights = self.weight.clamp(min=0)
            if parameters["gamma"]!=0:
                pweights +=pweights.clamp(min=0)*parameters["gamma"]
            pZ = F.linear(pAij, pweights, bias=None).clamp(min=0)
            pZ += torch.sign(pZ)*parameters["epsilon"]  #chapter 7 equation 7.5  zk = ε + ∑0,j aj · wjk
            if concepts is not None: # Algorithm 2 Concept Relevance Propagation page-46 chapter 7
                self.conceptmask.register_concept_channel(concepts,pZ.shape)
                for concept_index,conceptmask in self.conceptmask.mask.items(): # loop over generated masks θl={θ1,θ2..θn}.
                    for index,conceptrelevance in fowardlayerrelevance.items(): #looped over existing masks from forward layers
                        concept_name=str(index)+"_"+str(concept_index) # generating new name for the concept
                        
                        sensitivity=((conceptrelevance*conceptmask) / (pZ+(pZ == 0).float())) # chapter 7 ,equation 7.6 sk = δjl*Rk/zk
                        pR = F.linear(sensitivity, pweights.t(), bias=None) #equation 7.7 ck = ∑  wjk · sk
                        layerrelevance=pR*pAij  #equation 7.7 Rj(x|θl)=aj*ck
                        outputconceptrelevance[concept_name]=layerrelevance
                self.conceptmask.deregister_concept_channel() # removed the mask to avoid memory overflow
            if conceptestimated: #if concepts channels are identified using rel max 
                for index,conceptrelevance in fowardlayerrelevance.items():
                    estimated_indicies=self.concept_index[index].keys()
                    
                    self.conceptmask.register_batch_concept_channel(estimated_indicies,pZ.shape)
                    for concept_index,conceptmask in self.conceptmask.mask.items():
                        concept_name=str(index)+"_"+str(concept_index)
                        
                        sensitivity=((conceptrelevance*conceptmask) / (pZ+(pZ == 0).float()))
                        pR = F.linear(sensitivity, pweights.t(), bias=None)
                        layerrelevance=pR*pAij
                        outputconceptrelevance[concept_name]=layerrelevance
                    self.conceptmask.deregister_concept_channel()
                return outputconceptrelevance   
            else:  # if there are not concept provided, it should just pass the forward relevance masks
                for index,conceptrelevance in fowardlayerrelevance.items():
                    sensitivity=(conceptrelevance / (pZ+(pZ == 0).float()))
                    pR = F.linear(sensitivity, pweights.t(), bias=None)
                    layerrelevance=pR*pAij
                    outputconceptrelevance[index]=layerrelevance
                return outputconceptrelevance
        elif rule=="lrpalphabeta":  #implemented the rule based on the Table 3.1 in chapeter 3 perform concept relevance using alpha beta
            # seperating negative and posititve activation, weights and calculating pre-activations
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
                        
                        outputconceptrelevance[index]=layerrelevance 
                return outputconceptrelevance
        else:
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
