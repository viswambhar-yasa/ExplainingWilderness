import torch
import torch.nn as nn

class IndexOutOfRangeError(Exception):
    pass

class RelevanceConcepts:
    def init(self):
        super(RelevanceConcepts,self).__init__()

    def register_channels(self,channels):
        self.channels=channels
        self.channel_index= torch.arange(0, channels)
        self.mask={}


    def register_concept_channel(self,indices,input_shape):
        if len(self.mask) > 0:
            self.deregister_concept_channel(self)
        for index in indices:
            if 0 <= index < self.channels:
                conceptmask = torch.zeros(input_shape)
                conceptmask[:,index,...]=1
                self.mask[index]=conceptmask
            else:
                raise IndexOutOfRangeError(f"Index {index} is outside the range of valid indices.")
        pass

    def register_batch_concept_channel(self,indices,input_shape):
        if len(self.mask) > 0:
            self.deregister_concept_channel(self)
        conceptmask = torch.zeros(input_shape)
        for batchno,row in enumerate(indices,0):
            batch_index=row.squeeze().tolist()
            if isinstance(batch_index,list):
                for index in batch_index:
                    if 0 <= index < self.channels:
                        conceptmask = torch.zeros(input_shape)
                        conceptmask[:,index,...]=1
                        self.mask["bi"+str(batchno)+"_"+str(index)]=conceptmask
                    else:
                        raise IndexOutOfRangeError(f"Index {index} is outside the range of valid indices.")
            else:
                if 0 <= batch_index < self.channels:
                    conceptmask = torch.zeros(input_shape)
                    conceptmask[:,batch_index,...]=1
                    self.mask[batch_index]=conceptmask
                else:
                    raise IndexOutOfRangeError(f"Index {index} is outside the range of valid indices.")
        pass

    def deregister_concept_channel(self):
        self.mask={}

    def max_relevance_concept(self,channelrelevance,topknum=2):
        relevance_concept_index={}
        for index,conceptrelevance in channelrelevance.items():
            conceptrelevance_shape=conceptrelevance.shape
            conceptrelevance_sum = conceptrelevance.view(*conceptrelevance_shape[:2], -1).sum(-1)
            max_values, indices = torch.topk(conceptrelevance_sum, k=topknum, dim=1)
            max_relevance_concepts = {indices: max_values}
            relevance_concept_index[index]=max_relevance_concepts
        return relevance_concept_index

    #def activation_maximization(self,activation,mask,topknum=2):
    #    activation_concept_index={}
    #    for index,conceptmask in mask.items():
    #        input_shape=conceptmask.shape
    #        activationmask=activation*conceptmask
    #        activationmask=activationmask.view(*activationmask.shape[:2], -1)
    #        max_values, indices = torch.topk(activationmask, k=topknum)
    #        max_activation = {str(index)+"_"+str(ind.item()): max_val.item() for ind,max_val in zip(indices[0],max_values[0])}
    #        activation_concept_index[index]=max_activation
    #        self.register_concept_channel(max_activation.keys(),input_shape)
    #    return activation_concept_index
