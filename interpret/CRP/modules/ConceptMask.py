# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Contains Class which perform concept isolation by selecting or identify channels of interest within the layer
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 


import torch

class IndexOutOfRangeError(Exception):
    pass

class RelevanceConcepts:
    """
    A class for managing concept channels and calculating maximum relevance concepts.

    Attributes:
        channels (int): The total number of channels.
        channel_index (torch.Tensor): A tensor with values from 0 to channels-1.
        mask (dict): A dictionary to store binary mask tensors for concept channels.

    Methods:
        register_channels(channels): Initializes the channels attribute and creates the channel_index tensor.
        register_concept_channel(indices, input_shape): Registers concept channels by creating binary mask tensors.
        register_batch_concept_channel(indices, input_shape): Registers batch concept channels by creating binary mask tensors.
        deregister_concept_channel(): Clears the mask dictionary.
        max_relevance_concept(channelrelevance, topknum=2): Calculates the top relevance concepts for each channel.

    Example Usage:
        rc = RelevanceConcepts()
        rc.register_channels(10)
        rc.register_concept_channel([0, 2, 4], (3, 3, 3))
        rc.register_batch_concept_channel([[1, 3], [5, 7]], (3, 3, 3))
        rc.max_relevance_concept(channel_relevance, topknum=2)
    """

    def __init__(self):
        super(RelevanceConcepts, self).__init__()

    def register_channels(self, channels):
        """
        Initializes the channels attribute and creates the channel_index tensor.

        Args:
            channels (int): The total number of channels.
        """
        self.channels = channels
        self.channel_index = torch.arange(0, channels) # creates a list of channel indices 
        self.mask = {}

    def register_concept_channel(self, indices, input_shape):
        """
        Registers concept channels by creating binary mask tensors .

        Args:
            indices (list): The indices of the concept channels to be registered.
            input_shape (tuple): The shape of the input data.
        """
        if len(self.mask) > 0:
            self.deregister_concept_channel(self)
        for index in indices:
            if 0 <= index < self.channels:
                conceptmask = torch.zeros(input_shape)
                conceptmask[:, index, ...] = 1 # Chapter 7, the equation 7.7 requires a δjl kronecker delta 
                self.mask[index] = conceptmask # multiple masks can be stored to perform concept seperation
            else:
                raise IndexOutOfRangeError(f"Index {index} is outside the range of valid indices.")
        pass

    def register_batch_concept_channel(self, indices, input_shape):
        """
        Registers batch concept channels by creating binary mask tensors if each image in the batch need to have it's own concepts .

        Args:
            indices (list): The indices of the concept channels to be registered.
            input_shape (tuple): The shape of the input data.
        """
        if len(self.mask) > 0:
            self.deregister_concept_channel(self)
        conceptmask = torch.zeros(input_shape)
        for batchno, row in enumerate(indices, 0):
            batch_index = row.squeeze().tolist()
            if isinstance(batch_index, list):
                for index in batch_index:
                    if 0 <= index < self.channels:
                        conceptmask = torch.zeros(input_shape)
                        conceptmask[:, index, ...] = 1 # Chapter 7, the equation 7.7 requires a δjl kronecker delta 
                        self.mask["bi" + str(batchno) + "_" + str(index)] = conceptmask # multiple masks can be stored to perform concept seperation
                    else:
                        raise IndexOutOfRangeError(f"Index {index} is outside the range of valid indices.")
            else:
                if 0 <= batch_index < self.channels:
                    conceptmask = torch.zeros(input_shape)
                    conceptmask[:, batch_index, ...] = 1 
                    self.mask[batch_index] = conceptmask
                else:
                    raise IndexOutOfRangeError(f"Index {index} is outside the range of valid indices.")
        pass

    def deregister_concept_channel(self):
        """
        Clears the mask dictionary, After the concept seperation, we delete the mask to avoid memory overflow.
        """
        self.mask = {}

    def max_relevance_concept(self, channelrelevance, topknum=2,largest=True):
        """
        Calculates the top relevance concepts for each channel. 

        Args:
            channelrelevance (dict): A dictionary where the keys are channel indices and the values are relevance scores.
            topknum (int): The number of top concepts to be returned.

        Returns:
            dict: A dictionary where the keys are channel indices and the values are dictionaries containing the top relevance concepts and their scores.
        """
        relevance_concept_index = {} # stores the relavance indices of each layer
        for index, conceptrelevance in channelrelevance.items():
            conceptrelevance_shape = conceptrelevance.shape
            conceptrelevance_sum = conceptrelevance.view(*conceptrelevance_shape[:2], -1).sum(-1) # performing summation based on chapter 3,  equation 3.25  
            max_values, indices = torch.topk(conceptrelevance_sum, k=topknum,largest=largest, dim=1) #Chapter 7 , equation 7.9 B = {b1, . . . , bn} = argsort Ril−1 (x | θc)
            max_relevance_concepts = {indices: max_values}
            relevance_concept_index[index] = max_relevance_concepts
        return relevance_concept_index