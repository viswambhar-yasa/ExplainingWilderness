# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: The ConceptRelevance class contain method which perform concept disentanglement and create plot of relevance heatmap based on selecting type
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import os
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from zennit.image import imsave
from crp.cache import ImageCache
from crp.helper import get_layer_names
from crp.concepts import ChannelConcept
from crp.graph import trace_model_graph
from torchvision.utils import save_image
from crp.helper import get_output_shapes
from crp.attribution import AttributionGraph
from crp.visualization import FeatureVisualization
from crp.image import vis_opaque_img,vis_img_heatmap,imgify

from interpret.concept.conceptplots import save_to_image,combine_images,save_grid,plot_concepts
from interpret.concept.conceptpropagation import ConceptVisualization,ConceptRelevanceAttribute,get_composite,get_relevance_function,get_layer_types

class ConceptRelevance:
    def __init__(self, model, device=None, overwrite_data_grad=True, no_param_grad=True, layer_type=[nn.Conv2d, nn.Linear], custom_mask=None) -> None:
        """
        Initializes an object of the ConceptRelevance class.

        Args:
            model (object): The model object that will be used for concept relevance analysis.
            device (str, optional): The device on which the analysis will be performed. Defaults to 'cuda' if available, otherwise 'cpu'.
            overwrite_data_grad (bool, optional): Whether to overwrite the gradients of the input data. Defaults to True.
            no_param_grad (bool, optional): Whether to disable gradients for model parameters. Defaults to True.
            layer_type (list, optional): The types of layers to consider for concept relevance analysis. Defaults to [nn.Conv2d, nn.Linear].
            custom_mask (object, optional): A custom mask to be used for concept relevance analysis. Defaults to None.
        """
        self.__dict__.clear()
        self.model = copy.deepcopy(model) #model is copied, so multiple copies are not created
        self.model.eval()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.attribute = ConceptRelevanceAttribute(self.model, self.device, overwrite_data_grad, no_param_grad) # conditional relevance class is assigned to the model to generate a graph network 
        self.cc = ChannelConcept() # relevance of channels of all layer are stored here
        self.register_conceptMasks(layer_type, custom_mask) #if there are any masks, we assign them so relevance flows throught the required area
        self.layertype_map = get_layer_types(self.model) #get the layer map
        self.fv =None
        pass
    
    def register_conceptMasks(self, layer_type, custom_mask):
        """
        Registers concept masks for specific layer types in the model.

        Args:
            layer_type (list): A list of layer types to consider for concept relevance analysis.
            custom_mask (object, optional): A custom mask to be used for concept relevance analysis. Defaults to None.

        Returns:
            None
        """
        self.layer_names = get_layer_names(self.model, layer_type)
        self.layer_map = {layer_name: custom_mask if custom_mask is not None else self.cc for layer_name in self.layer_names}


    def init_lrp(self, data, compositename="epsilon", canonizerstype="vgg", output_type="softmax"):
        """
        Initializes the concept relevance analysis by computing the relevance heatmaps for a given input data.

        Args:
            data (torch.Tensor): The input data for which the relevance heatmaps will be computed.
            compositename (str, optional): The name of the composite function. Defaults to "epsilon".
            canonizerstype (str, optional): The type of canonizers. Defaults to "vgg".
            output_type (str, optional): The type of output. Defaults to "softmax".

        Returns:
            tuple: A tuple containing the relevance heatmaps, conditions used for the analysis, and the predicted class index.

        Example:
            # Initialize the ConceptRelevance object
            cr = ConceptRelevance(model)

            # Load input data
            data = ...

            # Compute relevance heatmaps
            heatmaps, conditions, prediction = cr.init_lrp(data)
        """
        init_rel = get_relevance_function(output_type) # the initial output type is selected
        composite = get_composite(compositename, canonizerstype) # composite contains the specific propagtion rule 
        prediction = torch.argmax(torch.softmax(self.model(data), dim=-1), dim=-1) #identifying the predicted value
        unq_prediction = torch.unique(prediction) # for batch of images, we select unique list 
        if unq_prediction.shape[0] == 1:
            index = unq_prediction.detach().numpy()
        else:
            index = unq_prediction.squeeze().detach().numpy()
        conditions = [{"y": list(index)}] #creating condition list
        heatmaps, _, _, _ = self.attribute(data, conditions, composite, init_rel=init_rel) # performing relevance flow and generating heatmap
        return heatmaps, conditions, prediction

    def layer_relevance(self, data, n_classes=2, compositename="epsilonplus", canonizerstype="vgg", output_type="max", saveplot=False, filepath=".", filename="relevance_heatmap", level=2.5):
        """
        Computes the relevance heatmaps for a given input data.

        Args:
            data (torch.Tensor): The input data for which the relevance heatmaps will be computed.
            n_classes (int or list): The number of classes or a list of class indices to consider for relevance analysis.
            compositename (str): The name of the composite function to use for relevance propagation.
            canonizerstype (str): The type of canonizers to use for relevance propagation.
            output_type (str): The type of output to consider for relevance analysis.
            saveplot (bool): Whether to save the heatmaps as plots.
            filepath (str): The path where the heatmaps will be saved.
            filename (str): The name of the heatmaps file.
            level (float): The threshold level for generating the heatmaps.

        Returns:
            None

        Example Usage:
            cr = ConceptRelevance(model)
            data = ...
            cr.layer_relevance(data, n_classes=2, compositename="epsilonplus", canonizerstype="vgg", output_type="max", saveplot=True, filepath=".", filename="relevance_heatmap", level=2.5)
        """
        if not data.requires_grad:
            data.requires_grad = True # data should be able to generate gradients
        conditions = []
        if isinstance(n_classes, int):
            conditions = [{"y": i} for i in range(n_classes)]
            n_rows = n_classes
        elif isinstance(n_classes, list):
            conditions = [{"y": i} for i in n_classes]
            n_rows = len(n_classes)
        init_rel = get_relevance_function(output_type)
        composite = get_composite(compositename, canonizerstype)
        heatmaps, _, relevance, _ = self.attribute(data, conditions, composite, init_rel=init_rel)
        if saveplot: # the heatmaps are saved as plots
            save_to_image(heatmaps, data, filepath, filename, n_rows, relevance, level)
        pass
    
      
    def compute_relevance_maximization(self, relevance, condlayernames, relevance_type="abs", topk_c=5):
        """
        Calculates the relevance maximization for each layer in the model.

        Args:
            relevance (dict): A dictionary containing the relevance values for each layer in the model.
            condlayernames (list): A list of layer names to consider for relevance maximization.
            relevance_type (str, optional): The type of relevance to consider ("abs", "negative", or "positive"). Defaults to "abs".
            topk_c (int, optional): The number of top-k channels to select. Defaults to 5.

        Returns:
            dict: A dictionary containing the top-k indices and relevances for each layer in the model.
        """
        "if you get any error remove the last output record layer or input layer as they only have 3 and 2 channels"
        toprelevance_list={}#contains layername and their channel index with relevance
        for layer_name in condlayernames: # looping over the selected layers
            rel_dict = []

            for i in range(len(relevance[layer_name])):
                if len(relevance[layer_name].shape) == 4:  # For 4D tensor
                            imgrelevance = relevance[layer_name][i, :, :, :].unsqueeze(dim=0)
                else:
                            imgrelevance = relevance[layer_name][i].unsqueeze(dim=0)

                channel_rels = self.cc.attribute(imgrelevance, abs_norm=True).squeeze(dim=0)
                if channel_rels.shape[0]<topk_c:
                    topk_c=channel_rels.shape[0]
                if relevance_type == "abs":
                    topk_indices = torch.topk(channel_rels.abs(), topk_c, dim=0).indices
                elif relevance_type == "negative":
                    topk_indices = torch.topk(channel_rels, topk_c, dim=0, largest=False).indices
                else:  # Default is "positive" relevance
                    topk_indices = torch.topk(channel_rels, topk_c, dim=0).indices
                topk_indices = topk_indices.detach().cpu().numpy()
                # Efficiently summing the relevances for the top-k channels
                topk_rel = (channel_rels[topk_indices] * 100).tolist()
                rel_dict.append((list(topk_indices), topk_rel))

                toprelevance_list[layer_name] = rel_dict
            
        return toprelevance_list
    
    def build_concept_disentangle(self, data, record_layers=None):
        """
        Builds an attribution graph for concept disentanglement. 
        The concept disentanglement requires running the backpropagation recussively. As recussive is computationally expensive.
        We use graph network to generate the concept disentanglement.

        Args:
            data: The input data for which the attribution graph will be built.
            record_layers (optional): A list of layer names to record. If not provided, all layers in the model will be recorded.

        Returns:
            None

        Summary:
        The `build_concept_disentangle` method is used to build an attribution graph for concept disentanglement. It takes in the input data and a list of layer names to record, and then traces the model graph using the `trace_model_graph` function. The resulting graph is used to create an `AttributionGraph` object, which is stored in the `ConceptRelevance` class.

        Example Usage:
        ```python
        cr = ConceptRelevance(model)
        data = ...
        cr.build_concept_disentangle(data, record_layers=["layer1", "layer2"])
        ```

        Code Analysis:
        - If `record_layers` is not provided, it is set to the default value of `self.layer_names` and `layer_map` is set to `self.layer_map`.
        - The `trace_model_graph` function is called to trace the model graph using the input data and the `record_layers`.
        - The resulting graph is used to create an `AttributionGraph` object, which is stored in `self.ConceptsGraph`.
        """
        if record_layers is None:# the layer which have to be be converted to graph network are computed
            record_layers = self.layer_names
            layer_map = self.layer_map
        else:
            layer_map = {name: self.cc for name in record_layers}
        graph = trace_model_graph(self.model, data, record_layers) # creates graph network
        self.ConceptsGraph = AttributionGraph(self.attribute, graph, layer_map) # assigning the graph to the conditional relevance class
        pass
        
    def compute_concept_disentangle(self, data, channel_index, conceptlayer, higher_concept_index=1, compositename="epsilonplus", canonizerstype="vgg", record_layers=None, width=[3, 1], build=True, abs_norm=True):
        """
        Compute the concept disentanglement for a given input data.

        Args:
            data (numpy.ndarray): The input data for which the concept disentanglement will be computed.
            channel_index (int): The index of the channel to consider for concept disentanglement.
            conceptlayer (str): The layer name for which the concept disentanglement will be computed.
            higher_concept_index (int, optional): The index of the higher-level concept to consider final prediction. Defaults to 1.
            compositename (str, optional): The name of the composite function to use for concept propagation. Defaults to "epsilonplus".
            canonizerstype (str, optional): The type of canonizers to use for concept propagation. Defaults to "vgg".
            record_layers (list, optional): The layers to record for concept disentanglement. Defaults to None.
            width (list, optional): The width of the concept disentanglement graph. Defaults to [3, 1].
            build (bool, optional): Whether to build the concept disentanglement graph. Defaults to True.
            abs_norm (bool, optional): Whether to normalize the relevance values. Defaults to True.

        Returns:
            tuple: A tuple containing two dictionaries:
                - nodes_dict (dict): A dictionary containing the concept nodes for each layer.
                - layer_connections (dict): A dictionary containing the connections between layers and their corresponding concept nodes.
        """
        if not data.requires_grad:
            data.requires_grad = True 
        composite = get_composite(compositename, canonizerstype) # get propagation rule 
        if build:
            self.build_concept_disentangle(data, record_layers) # build the grpah network
        # performing concept disentanglement 
        nodes, connections = self.ConceptsGraph(data, composite, channel_index, conceptlayer, higher_concept_index, width=width, abs_norm=abs_norm)
        nodes_dict = {k: [] for k, _ in nodes}
        # creating a better dictionary which has a relationship of channels and relevance flow
        for k, v in nodes:
            nodes_dict[k].append(v)
        layer_connections = {}
        for key, value in connections.items():
            layer_name, index = key
            for x in value:
                feature_layer, channel_index, rel = x
                try:
                    layer_connections[(feature_layer, layer_name + ":" + feature_layer)].append((channel_index, str(index) + ":" + str(channel_index), rel))
                except KeyError:
                    layer_connections[(feature_layer, layer_name + ":" + feature_layer)] = [(channel_index, str(index) + ":" + str(channel_index), rel)]
        layer_connections[conceptlayer] = nodes_dict[conceptlayer]
        return nodes_dict, layer_connections
    
    def visualize_concept_disentangle(self, data,condition, channel_index, conceptlayer, higher_concept_index=1, compositename="epsilonplus", canonizerstype="vgg", output_type="max", filename="concepts", filepath="./conceptdisentangle", record_layers=None, width=[3,1], build=True, abs_norm=True, level=2, figsize=(25,15)):
        """
        Visualizes the concept disentanglement process in the ConceptRelevance class by generating heatmaps and saving them as images.

        Args:
            data (torch.Tensor): The input data for which the relevance heatmaps will be computed.
            channel_index (int): The index of the channel to visualize.
            conceptlayer (str): The layer at which the concept is located.
            higher_concept_index (int, optional): The index of the higher-level concept to consider. Defaults to 1.
            compositename (str, optional): The name of the composite function to use for relevance propagation. Defaults to "epsilonplus".
            canonizerstype (str, optional): The type of canonizers to use for relevance propagation. Defaults to "vgg".
            output_type (str, optional): The type of output to consider for relevance analysis. Defaults to "max".
            filename (str, optional): The name of the heatmaps file. Defaults to "concepts".
            filepath (str, optional): The path where the heatmaps will be saved. Defaults to "./conceptdisentangle".
            record_layers (list, optional): The layers to record the concept disentanglement process for. If None, all layers will be recorded. Defaults to None.
            width (list, optional): The width of the concept disentanglement plot. Defaults to [3,1].
            build (bool, optional): Whether to build the concept disentanglement plot. Defaults to True.
            abs_norm (bool, optional): Whether to normalize the relevance values. Defaults to True.
            level (int, optional): The threshold level for generating the heatmaps. Defaults to 2.
            figsize (tuple, optional): The size of the concept disentanglement plot. Defaults to (25,15).

        Returns:
            None
        """
        if record_layers is not None:
            condlayernames = record_layers
        else:
            condlayernames = self.layer_names
        prediction = torch.argmax(torch.softmax(self.model(data), dim=-1), dim=-1)
        if condition is None:
            unq_prediction = torch.unique(prediction)
            if unq_prediction.shape[0] == 1:
                index = unq_prediction.detach().numpy()
            else:
                index = unq_prediction.squeeze().detach().numpy()
            condition = [{"y": list(index)}]
        nodes, connections = self.compute_concept_disentangle(data, channel_index, conceptlayer, higher_concept_index, compositename, canonizerstype, condlayernames, width, build, abs_norm)
        heatmap, relevance, _ = self.conditional_relevance(data, condition, compositename, canonizerstype, output_type, condlayernames)
        # plotting the heatmaps of all the concepts
        heatmap_filename = filename + "_lrp_" + str(list(prediction.detach().numpy())) + ".png"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        heatmap_path = os.path.join(filepath, heatmap_filename)
        imsave(heatmap_path, F.pad(heatmap, (2, 2, 2, 2), mode='constant', value=0), cmap="france", level=level, grid=(data.shape[0], 1), symmetric=True)
        image_path = os.path.join(filepath, filename + "_data.png")
        save_image(F.pad(data, (2, 2, 2, 2), mode='constant', value=0), image_path, nrow=1, padding=0)
        combine_images(heatmap_path, image_path, filepath, filename + "_lrp_" + str(list(prediction.detach().numpy())))
        init_rel = get_relevance_function(output_type)
        composite = get_composite(compositename, canonizerstype)
        for keys, value in connections.items():
            for x in value:
                if isinstance(x, tuple):
                    layer = keys[0]
                    channel_index = x[0]
                    conditional_channel = x[0]
                else:
                    layer = keys
                    channel_index = x
                    conditional_channel = "-"
                if not isinstance(channel_index, list):
                    channel_index = [channel_index]
                    n = len(channel_index)
                batchheatmap = {}
                for image_index in tqdm(range(0, data.shape[0]), desc=f'Processing {layer}'):
                    image = data[image_index, :, :, :].unsqueeze(0)
                    conditions = [{layer: [id], 'y': prediction[image_index]} for id in channel_index]
                    condchannel_hm, _, _, _ = self.attribute(image, conditions, composite, init_rel=init_rel)
                    batchheatmap[image_index] = (conditional_channel, imgify(F.pad(condchannel_hm, (5, 5, 5, 5), mode='constant', value=0), cmap="france", level=level, grid=(1, n), symmetric=True))
                heatmap_path = os.path.join(filepath, filename + "_" + layer + ".png")
                plot_concepts(batchheatmap, heatmap_path, layer, figsize)
        dict_filename = os.path.join(filepath, filename + "channel_index.json")
        torch.save(relevance, dict_filename)
        pass
            
        
    def conditional_relevance(self, data, condition, compositename="epsilonplus", canonizerstype="vgg", output_type="max", record_layer: list = None):
        """
        Compute the relevance heatmaps for a given input data based on specific conditions.

        Args:
            data: The input data for which the relevance heatmaps will be computed.
            condition: The condition or conditions to be applied during the relevance analysis.
            compositename (optional): The name of the composite function to use for relevance propagation. Defaults to "epsilonplus".
            canonizerstype (optional): The type of canonizers to use for relevance propagation. Defaults to "vgg".
            output_type (optional): The type of output to consider for relevance analysis. Defaults to "max".
            record_layer (optional): A list of layer names to record during the relevance analysis. Defaults to None.

        Returns:
            heatmaps: The computed relevance heatmaps for the input data.
            relevance: The relevance values for each layer in the model.
            prediction: The predicted class index for the input data.
        """
        if not data.requires_grad:
            data.requires_grad = True
        init_rel = get_relevance_function(output_type)
        composite = get_composite(compositename, canonizerstype, self.layertype_map)
        heatmaps, _, relevance, prediction = self.attribute(data, condition, composite, record_layer=record_layer, init_rel=init_rel)
        return heatmaps, relevance, prediction
    

    def visualize_concepts(self, data, condition=None, compositename="epsilon", canonizerstype="vgg", output_type="max", record_layer=None, relevance_type="positive", topk_c=5, filepath=".", filename="concepts", cmap="hot", symmetric=False, level=2, figsize=(25, 15)):
        """
        Generate and save visualizations of concept relevance analysis.

        Args:
            data (torch.Tensor): The input data for which the relevance heatmaps will be computed and visualized.
            condition (dict, optional): The condition for relevance analysis. If not provided, the predicted class index will be used.
            compositename (str, optional): The name of the composite function to use for relevance propagation. Defaults to "epsilon".
            canonizerstype (str, optional): The type of canonizers to use for relevance propagation. Defaults to "vgg".
            output_type (str, optional): The type of output to consider for relevance analysis. Defaults to "max".
            record_layer (list, optional): The list of layer names to record relevance for. If not provided, all layers will be considered.
            relevance_type (str, optional): The type of relevance to consider for maximization. Defaults to "positive".
            topk_c (int, optional): The number of top concepts to visualize for each layer. Defaults to 5.
            filepath (str, optional): The path where the visualizations will be saved. Defaults to current directory.
            filename (str, optional): The base name of the visualizations. Defaults to "concepts".
            cmap (str, optional): The colormap to use for visualizations. Defaults to "hot".
            symmetric (bool, optional): Whether to use symmetric color mapping. Defaults to False.
            level (int, optional): The threshold level for generating the visualizations. Defaults to 2.
            figsize (tuple, optional): The size of the figure for visualizations. Defaults to (25, 15).

        Returns:
            Visualizations: Heatmap images showing the relevance of concepts for each layer.
            Relevance values: A dictionary containing the relevance values for each layer and concept.
        """
        if record_layer is not None:
            condlayernames = record_layer
        else:
            condlayernames = self.layer_names
        prediction = torch.argmax(torch.softmax(self.model(data), dim=-1), dim=-1)
        if condition is None:
            unq_prediction = torch.unique(prediction)
            if unq_prediction.shape[0] == 1:
                index = unq_prediction.detach().numpy()
            else:
                index = unq_prediction.squeeze().detach().numpy()
            condition = [{"y": list(index)}]
        heatmap, relevance, _ = self.conditional_relevance(data, condition, compositename, canonizerstype, output_type, condlayernames)
        heatmap_filename = filename + "_lrp_" + str(list(prediction.detach().to("cpu").numpy())) + ".png"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        heatmap_path = os.path.join(filepath, heatmap_filename)
        imsave(heatmap_path, F.pad(heatmap, (2, 2, 2, 2), mode='constant', value=0).detach().to("cpu").numpy(), cmap="france", level=level, grid=(data.shape[0], 1), symmetric=True)
        image_path = os.path.join(filepath, filename + "_data.png")
        save_image(F.pad(data, (2, 2, 2, 2), mode='constant', value=0), image_path, nrow=1, padding=0)
        combine_images(heatmap_path, image_path, filepath, filename + "_lrp_" + str(list(prediction.detach().to("cpu").numpy())))
        # max relevance of the selected layer are calculated
        toprelevance_list = self.compute_relevance_maximization(relevance, condlayernames, relevance_type, topk_c)
        init_rel = get_relevance_function(output_type)
        composite = get_composite(compositename, canonizerstype)
        # the condition concept of the respective top max channels are calculated and plotted
        for layer_name in condlayernames:
            batchheatmap = {}
            for image_index in tqdm(range(0, data.shape[0]), desc=f'Processing {layer_name}'):
                image = data[image_index, :, :, :].unsqueeze(0)
                channel_index = list(toprelevance_list[layer_name][image_index][0])
                conditions = [{layer_name: [id], 'y': prediction[image_index]} for id in channel_index]
                condchannel_hm, _, _, _ = self.attribute(image, conditions, composite, init_rel=init_rel)
                batchheatmap[image_index] = (channel_index, imgify(F.pad(condchannel_hm, (5, 5, 5, 5), mode='constant', value=0).detach().to("cpu").numpy(), cmap=cmap, level=level, grid=(1, topk_c), symmetric=symmetric))
            heatmap_path = os.path.join(filepath, filename + "_" + layer_name + ".png")
            plot_concepts(batchheatmap, heatmap_path, layer_name, figsize)
        dict_filename = os.path.join(filepath, filename + "channel_index.pkl")
        torch.save(relevance, dict_filename) # saving the relevance of init lrp for analysis
        pass
    
    
    def build_reference_images(self, dataset, preprocessing, filesavepath, compositename="epsilonplusflat", canonizerstype="vgg", device="cpu", imagecache=True, imagecachefilepath="cache", max_target="max", build=False, batch_size=8, chkpoint=250):
        """
        Builds reference images for concept visualization based on the given dataset.

        Args:
            dataset (object): The dataset used to build the reference images.
            preprocessing (function): The preprocessing function applied to the dataset.
            filesavepath (str): The path where the reference images will be saved.
            compositename (str, optional): The name of the composite function to use for concept visualization. Defaults to "epsilonplusflat".
            canonizerstype (str, optional): The type of canonizers to use for concept visualization. Defaults to "vgg".
            device (str, optional): The device on which the visualization will be performed. Defaults to "cpu".
            imagecache (bool, optional): Whether to use an image cache for faster processing. Defaults to True.
            imagecachefilepath (str, optional): The file path for the image cache. Defaults to "cache".
            max_target (str, optional): The target for maximizing the relevance. Defaults to "max".
            build (bool, optional): Whether to build the reference images. Defaults to False.
            batch_size (int, optional): The batch size for processing the dataset. Defaults to 8.
            chkpoint (int, optional): The checkpoint interval for saving the reference images. Defaults to 250.

        Returns:
            None
        """
        if imagecache:
            cache = ImageCache(path=imagecachefilepath) # cache folder is created for quick analysis
        else:
            cache = None
        # to visualize the concept from dataset on the global scale, we assign the conditional class and other key parameter to calculate statistics for max relevance 
        self.fv = FeatureVisualization(self.attribute, dataset, self.layer_map, preprocess_fn=preprocessing, path=filesavepath, device=device, cache=cache, max_target=max_target)
        composite = get_composite(compositename, canonizerstype)
        if build: # running the analysis on the entire dataset 
            _ = self.fv.run(composite, 0, len(dataset), batch_size, chkpoint)
        pass


    def precompute_reference_images(self, compositename="epsilonplusflat", canonizerstype="vgg", relevance_range=(0,8), imagemode="relevance", receptivefield=False, batch=8):
        """
        Precomputes reference images for concept relevance analysis.

        Args:
            compositename (str): The name of the composite function to use for relevance propagation.
            canonizerstype (str): The type of canonizers to use for relevance propagation.
            relevance_range (tuple): The range of relevance values to consider for precomputing the reference images.
            imagemode (str): The mode for generating the reference images ("relevance" or "heatmap").
            receptivefield (bool): Whether to consider the receptive field for generating the reference images.
            batch (int): The batch size for precomputing the reference images.

        Returns:
            None

        Example Usage:
            cr = ConceptRelevance(model)
            cr.precompute_reference_images(compositename="epsilonplusflat", canonizerstype="vgg", relevance_range=(0,8), imagemode="relevance", receptivefield=False, batch=8)
        """
        # precomputing the reference images for each layer, to generate sample images for the prediction faster
        output_shape = get_output_shapes(self.model, self.fv.get_data_sample(0)[0], self.layer_names)
        composite = get_composite(compositename, canonizerstype)
        layer_id_map = {l_name: np.arange(0, out[0]) for l_name, out in output_shape.items()}
        self.fv.precompute_ref(layer_id_map, plot_list=[vis_opaque_img, vis_img_heatmap], mode=imagemode, r_range=relevance_range, composite=composite, rf=receptivefield, batch_size=batch, stats=False)
        pass


    def compute_reference_image(self, concepts_map, dataset, preprocessing, filesavepath, refimgsavepath=".z", compositename="epsilonplus", canonizerstype="vgg", device="cpu", imagecache=False, relevance_range=(0, 8), imagemode="relevance", cmap="france", plotfn=vis_img_heatmap, receptivefield=False, batch=8):
        """
        Compute reference images for specific concepts in a given dataset.

        Args:
            concepts_map (dict): A dictionary mapping layer names to a list of concept indices.
            dataset: The dataset used to compute the reference images.
            preprocessing: The preprocessing function applied to the dataset.
            filesavepath: The path where the reference images will be saved.
            refimgsavepath (str, optional): The path where the reference images for each concept will be saved. Defaults to ".z".
            compositename (str, optional): The name of the composite function used for relevance propagation. Defaults to "epsilonplus".
            canonizerstype (str, optional): The type of canonizers used for relevance propagation. Defaults to "vgg".
            device (str, optional): The device on which the computation will be performed. Defaults to "cpu".
            imagecache (bool, optional): Whether to use an image cache for faster computation. Defaults to False.
            relevance_range (tuple, optional): The range of relevance values used for visualization. Defaults to (0, 8).
            imagemode (str, optional): The mode used for generating the reference images. Defaults to "relevance".
            cmap (str, optional): The colormap used for visualization. Defaults to "france".
            plotfn: The function used for plotting the reference images.
            receptivefield (bool, optional): Whether to include the receptive field in the reference images. Defaults to False.
            batch (int, optional): The batch size used for computation. Defaults to 8.

        Returns:
            None
        """
        self.build_reference_images(dataset, preprocessing, filesavepath, compositename, canonizerstype, device, imagecache, build=False)
        composite = get_composite(compositename, canonizerstype)
        for layer_name, layer_indices in concepts_map.items():
            for batch_index in tqdm(range(0, len(layer_indices)), desc=f'Processing {layer_name}'):
                if isinstance(layer_indices[batch_index],tuple):
                    channelindex = layer_indices[batch_index][0]
                elif isinstance(layer_indices[batch_index],list):
                    channelindex = layer_indices[batch_index]
                # generating reference image with max relevance
                ref = self.fv.get_max_reference(channelindex, layer_name, imagemode, relevance_range, composite, receptivefield, plotfn, batch)
                path = os.path.join(refimgsavepath, "reference_images_" + str(batch_index))
                if not os.path.exists(path):
                    os.makedirs(path)
                rfimgpath = os.path.join(path, layer_name + "_rfimg.png")
                save_grid(ref_c=ref, cmap=cmap, filepath=rfimgpath)
        pass

    

    def concept_reference_images(self, data, dataset, preprocessing, condition=None, compositename="epsilonplus", canonizerstype="vgg", output_type="max", filesavepath="VGG16_bn_Sentinal", refimgsavepath="./concepts", imagecache=False, record_layer=None, relevance_type="abs", topk_c=5, relevance_range=(0,8), imagemode="relevance", plotfn=vis_img_heatmap, receptivefield=False, batch=8):
        """
        Calculate the initial relevance and register concepts for a given input data.
        Identify the channel index of each layer and compute the relevance maximization.
        Compute the reference images based on the top-k channels and save them.

        Args:
            data (Tensor): The input data for which the reference images will be computed.
            dataset (str): The dataset used for preprocessing.
            preprocessing (str): The preprocessing method applied to the dataset.
            condition (list, optional): The condition for relevance analysis. Defaults to None.
            compositename (str, optional): The name of the composite function used for relevance propagation. Defaults to "epsilonplus".
            canonizerstype (str, optional): The type of canonizers used for relevance propagation. Defaults to "vgg".
            output_type (str, optional): The type of output used for relevance analysis. Defaults to "max".
            filesavepath (str, optional): The path where the reference images will be saved. Defaults to "VGG16_bn_Sentinal".
            refimgsavepath (str, optional): The path where the concepts will be saved. Defaults to "./concepts".
            imagecache (bool, optional): Whether to use image caching. Defaults to False.
            record_layer (list, optional): The layers to consider for relevance maximization. Defaults to None.
            relevance_type (str, optional): The type of relevance to consider. Defaults to "abs".
            topk_c (int, optional): The number of top-k channels to select. Defaults to 5.
            relevance_range (tuple, optional): The range of relevance values. Defaults to (0,8).
            imagemode (str, optional): The mode of the reference images. Defaults to "relevance".
            plotfn (function, optional): The function used for plotting. Defaults to vis_img_heatmap.
            receptivefield (bool, optional): Whether to consider the receptive field. Defaults to False.
            batch (int, optional): The batch size. Defaults to 8.

        Returns:
            None
        """
        if record_layer is not None:
            condlayernames = record_layer
        else:
            condlayernames = self.layer_names
        prediction = torch.argmax(torch.softmax(self.model(data), dim=-1), dim=-1)
        if condition is None:
            unq_prediction = torch.unique(prediction)
            if unq_prediction.shape[0] == 1:
                index = unq_prediction.detach().to("cpu").numpy()
            else:
                index = unq_prediction.squeeze().detach().to("cpu").numpy()
            condition = [{"y": list(index)}]
        print("Calculating initial relevance and registering concepts")
        _, relevance, _ = self.conditional_relevance(data, condition, compositename, canonizerstype, output_type, condlayernames)
        print("Identifying the channel index of each layer")
        toprelevance_list = self.compute_relevance_maximization(relevance, condlayernames, relevance_type, topk_c)
        del relevance, data
        self.compute_reference_image(toprelevance_list, dataset, preprocessing, filesavepath, refimgsavepath, compositename=compositename, canonizerstype=canonizerstype, imagecache=imagecache, relevance_range=relevance_range, imagemode=imagemode, plotfn=plotfn, receptivefield=receptivefield, batch=batch)
        pass
        
    def layer_reference_image(self, layername, dataset, preprocessing, filesavepath,channelindex, compositename="epsilonplus", canonizerstype="vgg", device="cpu", imagecache=False, relevance_range=(0, 8), imagemode="relevance",max_target="max", plotfn=vis_img_heatmap, receptivefield=False,build=True, batch=8, chkpoint=50):
        """
        Compute reference images for specific concepts in a given dataset.

        Args:
            concepts_map (dict): A dictionary mapping layer names to a list of concept indices.
            dataset: The dataset used to compute the reference images.
            preprocessing: The preprocessing function applied to the dataset.
            filesavepath: The path where the reference images will be saved.
            refimgsavepath (str, optional): The path where the reference images for each concept will be saved. Defaults to ".z".
            compositename (str, optional): The name of the composite function used for relevance propagation. Defaults to "epsilonplus".
            canonizerstype (str, optional): The type of canonizers used for relevance propagation. Defaults to "vgg".
            device (str, optional): The device on which the computation will be performed. Defaults to "cpu".
            imagecache (bool, optional): Whether to use an image cache for faster computation. Defaults to False.
            relevance_range (tuple, optional): The range of relevance values used for visualization. Defaults to (0, 8).
            imagemode (str, optional): The mode used for generating the reference images. Defaults to "relevance".
            cmap (str, optional): The colormap used for visualization. Defaults to "france".
            plotfn: The function used for plotting the reference images.
            receptivefield (bool, optional): Whether to include the receptive field in the reference images. Defaults to False.
            batch (int, optional): The batch size used for computation. Defaults to 8.

        Returns:
            None
        """
        
        extracted_dict = {layername: self.layer_map[layername]}

        if self.fv is None:
            self.fv = FeatureVisualization(self.attribute, dataset, extracted_dict, preprocess_fn=preprocessing, path=filesavepath, device=device, cache="cache", max_target=max_target)
        composite = get_composite(compositename, canonizerstype)
        
        if build: # running the analysis on the entire dataset 
            _ = self.fv.run(composite, 0, len(dataset), batch, chkpoint)
        pass
        self.build_reference_images(dataset, preprocessing, filesavepath, compositename, canonizerstype, device, imagecache, build=False)
        composite = get_composite(compositename, canonizerstype)
        ref = self.fv.get_max_reference(channelindex, layername, imagemode, relevance_range, composite, receptivefield, plotfn, batch)
        return ref
