# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: implemented concept relevance propagation method for convolution  layer, the layer has additional functions without changing it's back propagation.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from interpret.CRP.modules.ConceptMask import RelevanceConcepts


class Conv2DCRP(nn.Conv2d):
    """
    The `Conv2DCRP` class is a subclass of `nn.Conv2d` in the PyTorch library. It is used for convolutional operations in deep learning models. This class extends the functionality of the base class by adding methods for interpreting the relevance of input features and concepts.

    Example Usage:
        # Create an instance of the Conv2DCRP class
        conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        conv_crp = Conv2DCRP(conv_layer)

        # Perform forward pass
        input = torch.randn(1, 3, 32, 32)
        output = conv_crp.forward(input)

        # Interpret the relevance of input features
        previous_layer_input = torch.randn(1, 3, 32, 32)
        forward_layer_relevance = {0: torch.randn(1, 64, 32, 32), 1: torch.randn(1, 64, 32, 32)}
        concepts = [0, 1, 2]
        concept_index_estimation = "relmax"
        top_num = 2
        rule = "lrp0"
        parameters = {"gamma": 0.5, "epsilon": 1e-7}
        relevance = conv_crp.interpet(previous_layer_input, forward_layer_relevance, concepts, concept_index_estimation, top_num, rule, parameters)

    Methods:
        - __init__(self, conv_layer: nn.Conv2d): Initializes the Conv2DCRP instance by copying the parameters of a given convolutional layer and initializing the conceptmask and concept_index attributes.
        - copy_parameters(self, module): Copies the parameters of a given module to the current instance.
        - forward(self, input: torch.tensor) -> torch.tensor: Performs the forward pass by calling the forward method of the base class.
        - interpet(self, previouslayer_input: torch.tensor, forwardlayerrelevance: dict, concepts=None, conceptindex_estimation=None, top_num=2, rule="lrp0", parameters={}) -> torch.tensor: Interprets the relevance of input features based on the given previous layer input, forward layer relevance, concepts, concept index estimation method, rule, and parameters. Returns the relevance of input features as a dictionary.

    Fields:
        - conceptmask: An instance of the RelevanceConcepts class used for masking concepts.
        - concept_index: A dictionary that stores the estimated concept indices for each input feature.
    """
    def __init__(self, conv_layer: nn.Conv2d):
        super().__init__(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, conv_layer.dilation, conv_layer.groups, conv_layer.bias is not None, conv_layer.padding_mode)
        self.copy_parameters(conv_layer)
        self.conceptmask=RelevanceConcepts() # creates a instance tostore the concept mask of the layer 
        self.conceptmask.register_channels(int(conv_layer.out_channels)) # regirstering the concept 
        self.concept_index=None

    def copy_parameters(self, module):
        """
        Copies the parameters of a given module to the current instance of the Conv2DCRP class.

        Args:
            module (torch.nn.Module): The module whose parameters need to be copied.

        Returns:
            None
        """
        with torch.no_grad():
            self.weight.data.copy_(module.weight.data)
            if self.bias is not None and module.bias is not None:
                self.bias.data.copy_(module.bias.data)
            self.kernel_size = module.kernel_size
            self.stride = module.stride
            self.padding = module.padding
            self.dilation = module.dilation
            self.groups = module.groups

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Performs the forward pass of the convolutional layer.

        Args:
            input (torch.tensor): The input tensor to the convolutional layer.

        Returns:
            torch.tensor: The output tensor of the convolutional layer.
        """
        return super(Conv2DCRP, self).forward(input)
    
    
    def interpet(self,previouslayer_input: torch.tensor, fowardlayerrelevance:dict,concepts=None,conceptindex_estimation=None,top_num=2,rule="lrp0",parameters={}) -> torch.tensor:
        """
        Interprets the relevance of input features and concepts in convolutional operations.

        :param previouslayer_input: A tensor representing the input to the previous layer.
        :param fowardlayerrelevance: A dictionary containing the relevance of the forward layer.
        :param concepts: Optional. A tensor representing the concepts.
        :param conceptindex_estimation: Optional. A string indicating the concept index estimation method.
        :param top_num: Optional. An integer indicating the number of top concepts to consider.
        :param rule: Optional. A string indicating the rule to use for interpretation.
        :param parameters: Optional. A dictionary containing additional parameters.

        :return: A dictionary containing the layer relevance for each concept.
        """
        Aij=previouslayer_input.data.requires_grad_(True)   
        outputconceptrelevance={}# dictionary to store relevance map, as each new concept need to be have it own path, we seperate them and store them in 
        conceptestimated=False 
        if conceptindex_estimation=="relmax":#if relevance maximum needs to be calculated for the forward layer, where the max relevance channels are selected
            self.concept_index=self.conceptmask.max_relevance_concept(fowardlayerrelevance,top_num)
            conceptestimated=True
        
        output_shape=Aij.shape
        first_key = next(iter(fowardlayerrelevance), None)  # Identifying existing concepts
        first_value = fowardlayerrelevance.get(first_key, None)
        relevance_output_shape = first_value.shape
        weight_shape = self.weight.shape
        #parameter required for back propagation calculation gradient calculation has showed significant consumption of memory, we implemented the backpropagation without gradients
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
        if rule=="zplus":   #implemented the rule based on the Table 3.1 in chapeter 3
            pAij=Aij.clamp(min=0)
            pweights = self.weight.clamp(min=0)
            if parameters["gamma"]!=0:
                pweights +=pweights.clamp(min=0)*parameters["gamma"]
            pZ=F.conv2d(pAij, pweights, None, self.stride, self.padding, self.dilation, self.groups).clamp(min=0)
            pZ += torch.sign(pZ)*parameters["epsilon"] #chapter 7 equation 7.5  zk = ε + ∑0,j aj · wjk
            if concepts is not None:# Algorithm 2 Concept Relevance Propagation page-46 chapter 7
                self.conceptmask.register_concept_channel(concepts,pZ.shape)
                for concept_index,conceptmask in self.conceptmask.mask.items(): # loop over generated masks θl={θ1,θ2..θn}.
                    for index,conceptrelevance in fowardlayerrelevance.items(): #looped over existing masks from forward layers
                        #print((conceptrelevance*conceptmask).sum())
                        concept_name=str(index)+"_"+str(concept_index)# generating new name for the concept
                        sensitivity=((conceptrelevance*conceptmask) / (pZ+(pZ == 0).float())) # chapter 7 ,equation 7.6 sk = δjl*Rk/zk
                        pR=F.conv_transpose2d(sensitivity, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)  #equation 7.7 ck = ∑  wjk · sk
                        layerrelevance=pR*pAij#equation 7.7 Rj(x|θl)=aj*ck
                        outputconceptrelevance[concept_name]=layerrelevance
                self.conceptmask.deregister_concept_channel() # removed the mask to avoid memory overflow
            if conceptestimated: #if concepts channels are identified using rel max 
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
            else: # if there are not concept provided, it should just pass the forward relevance masks
                for index,conceptrelevance in fowardlayerrelevance.items():
                        #print((conceptrelevance).sum())
                        sensitivity=(conceptrelevance / (pZ+(pZ == 0).float()))
                        pR=F.conv_transpose2d(sensitivity, pweights, None,self.stride, self.padding,output_padding,self.groups, self.dilation)
                        layerrelevance=pR*pAij
                        outputconceptrelevance[index]=layerrelevance
                return outputconceptrelevance
        elif rule=="alphabeta": #implemented the rule based on the Table 3.1 in chapeter 3 perform concept relevance using alpha beta
            # seperating negative and posititve activation, weights and calculating pre-activations
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
            # implementation of Algorithm 2 Concept Relevance Propagation page-46 chapter 7
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
            
        else: # default rule to extract concepts
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
