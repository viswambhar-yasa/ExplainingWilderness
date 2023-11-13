import copy
import torch
import numpy as np
import torch.nn as nn
from crp.cache import ImageCache
from crp.graph import trace_model_graph
from crp.concepts import ChannelConcept
from crp.attribution import AttributionGraph
from crp.image import vis_opaque_img,vis_img_heatmap
from crp.helper import get_layer_names,get_output_shapes
from interpet.concepts.customconcept import ConceptRelevanceAttribute,ConceptVisualization
from interpet.concepts.conceptsutils import compositetype,get_relevance_function


class ConceptRelevance:
    def __init__(self,model,device=None,overwrite_data_grad=True,no_param_grad=True,layer_type=[nn.Conv2d,nn.Linear],custom_mask=None) -> None:
        self.__dict__.clear()
        self.model=copy.deepcopy(model)
        self.model.eval()
        if device is None:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        self.attribute=ConceptRelevanceAttribute(self.model,self.device,overwrite_data_grad,no_param_grad)
        self.cc=ChannelConcept()
        self.register_conceptMasks(layer_type,custom_mask)
        pass

    def register_conceptMasks(self,layer_type,custom_mask):
        self.layer_names=get_layer_names(self.model,layer_type)
        self.layer_map = {layer_name: custom_mask if custom_mask is not None else self.cc for layer_name in self.layer_names}

    
    def concept_relevance_propagation(self,data,condition,compositename="epsilonplus",canonizerstype="vgg",output_type="max",record_layer:list=None):
        if not data.requires_grad:
            data.requires_grad=True
        init_rel=get_relevance_function(output_type)
        composite=compositetype(compositename,canonizerstype)
        heatmaps, _, relevance, prediction=self.attribute(data,condition,composite,record_layer=record_layer,init_rel=init_rel)
        return heatmaps,relevance,prediction
    
    def layerwise_propagation(self,data,target,compositename="epsilonplus",canonizerstype="vgg",output_type="softmax"):
        if not data.requires_grad:
            data.requires_grad=True
        init_rel=get_relevance_function(output_type)
        composite=compositetype(compositename,canonizerstype)
        if isinstance(target,int):
            conditions=[{"y":list(target)}]
        elif isinstance(target,list):
            conditions=[{"y":target}]
        else:
            prediction=torch.argmax(torch.softmax(self.model(data),dim=-1),dim=-1)
            unq_prediction=torch.unique(prediction)
            if unq_prediction.shape[0]==1:
                index=unq_prediction.detach().numpy()
            else:
                index=unq_prediction.squeeze().detach().numpy()
            conditions=[{"y":list(index)}]
        heatmaps, _, _, _ =self.attribute(data,conditions,composite,init_rel=init_rel)
        return heatmaps,conditions,prediction
    
    def compute_concept_maximization(self,relevance,condlayernames,maximization_type="abs",topk_c=5,mode="relevance"):
        toprelevance_list={}
        for layer_name in condlayernames:
            rel_dict=[]
            for i in range(0,len(relevance[layer_name])):
                if mode=="relevance":
                    channel_rels = self.cc.attribute(relevance[layer_name][i],layer_name=layer_name, abs_norm=True)
                else:
                    channel_rels = self.attribute.activations[layer_name].detach().cpu().flatten(start_dim=2).max(2)[0]
                    channel_rels = channel_rels / channel_rels.abs().sum(1)[:, None]
                if maximization_type=="abs":
                      topk = torch.topk(channel_rels[0].abs(), topk_c).indices.detach().cpu().numpy()
                elif maximization_type=="negative":
                    topk = torch.topk(channel_rels[0], topk_c,largest=False).indices.detach().cpu().numpy()
                else:
                    topk = torch.topk(channel_rels[0], topk_c).indices.detach().cpu().numpy()
                topk_rel = channel_rels[0][topk].detach().cpu().numpy()*100
                rel_dict.append((list(topk),list(topk_rel))) 
            toprelevance_list[layer_name]=rel_dict
        return toprelevance_list
    

    def build_concept_disentangle(self,data,record_layers=None):
        if record_layers is None:
            record_layers=self.layer_names
            layer_map=self.layer_map
        else:
            layer_map = {name: self.cc for name in record_layers}
        graph = trace_model_graph(self.model, data, record_layers)
        self.ConceptsGraph = AttributionGraph(self.attribute, graph, layer_map)
        pass

    def compute_concept_disentangle(self,data,channel_index,conceptlayer,higher_concept_index,compositename="epsilonplus",canonizerstype="vgg",record_layers=None,width=[3,1],build=True,abs_norm=True):
        if not data.requires_grad:
            data.requires_grad=True
        composite=compositetype(compositename,canonizerstype)
        if build:
            self.build_concept_disentangle(data,record_layers)
        nodes, connections = self.ConceptsGraph(data, composite, channel_index, conceptlayer, higher_concept_index, width=width, abs_norm=abs_norm)
        nodes_dict = {k: [] for k, _ in nodes}
        for k, v in nodes:
            nodes_dict[k].append(v)
        layer_connections={}
        for key,value in connections.items():
            layer_name,index=key
            for x in value:
                feature_layer,channel_index,rel=x
                try:
                    layer_connections[(feature_layer,layer_name+":"+feature_layer)].append((channel_index,str(index)+":"+str(channel_index),rel))
                except KeyError:
                    layer_connections[(feature_layer,layer_name+":"+feature_layer)] = [(channel_index,str(index)+":"+str(channel_index),rel)]
        layer_connections[conceptlayer]=nodes_dict[conceptlayer]
        return nodes_dict,layer_connections
    
    def compute_concepts(self,dataset,preprocessing,filesavepath,compositename="epsilonplusflat",canonizerstype="vgg",device="cpu",imagecache=True,imagecachefilepath="cache",max_target="max",build=False,batch_size=8,chkpoint=250):
        if imagecache:
            cache = ImageCache(path=imagecachefilepath)
        else:
            cache = None
        self.fv = ConceptVisualization(self.attribute, dataset,self.layer_map, preprocess_fn=preprocessing,path=filesavepath,device=device,cache=cache,max_target=max_target)
        composite=compositetype(compositename,canonizerstype)
        if build:
            _ = self.fv.run(composite, 0,len(dataset) , batch_size, chkpoint)
        pass
    
    def glocal_analysis(self,compositename="epsilonplusflat",canonizerstype="vgg",relevance_range=(0,8),imagemode="relevance",receptivefield=False,batch=8):
        output_shape = get_output_shapes(self.model, self.fv.get_data_sample(0)[0], self.layer_names)
        composite=compositetype(compositename,canonizerstype)
        layer_id_map = {l_name: np.arange(0, out[0]) for l_name, out in output_shape.items()}
        self.fv.precompute_ref(layer_id_map,  plot_list=[vis_opaque_img,vis_img_heatmap], mode=imagemode, r_range=relevance_range, composite=composite, rf=receptivefield, batch_size=batch, stats=False)
        pass
