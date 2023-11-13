import torch
import torch.nn as nn
from .modules.crplinear import LinearCRP
from .modules.crpconv2d import Conv2DCRP
from .modules.crpadaptivepool import AdaptiveAvgPool2dCRP
from .modules.crpmaxpool import MaxPool2dCRP

class CRPModel(nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model=model
        self.model.eval()
        self.converter(self.model)
        self.layer_inputs={}
        self.layer_relevance=[]
        self.layer_length=0
        print("Index of the layer required for concept registration /n")
        self.recursive_foward_hook(self.model,self.layer_length)
        
        self.lrp_default_parameters={
            "lrp0":{},
        "lrpepsilon": {"epsilon": 1, "gamma": 0},
        "lrpgamma": {"epsilon": 0.25, "gamma": 0.1},
        "lrpalpha1beta0": {"alpha": 1, "beta": 0,"epsilon": 1e-2, "gamma": 0},
        "lrpzplus": {"epsilon": 1e-2},
        "lrpalphabeta": {"epsilon": 1, "gamma": 0, "alpha": 2, "beta": 1}}
        
    def converter(self,model):
        for name, module in model.named_children():
            if isinstance(module,nn.Linear):
                
                setattr(model,name, LinearCRP(module))
            elif isinstance(module, nn.Conv2d):
                setattr(model,name, Conv2DCRP(module))
            elif isinstance(module, nn.MaxPool2d):
                setattr(model,name, MaxPool2dCRP(module))
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                setattr(model,name, AdaptiveAvgPool2dCRP(module))
            else:
                self.converter(module)
            

    def hook_function(self,module,input,output):
        self.layer_inputs[module]=input[0]    

    def recursive_foward_hook(self,model,index):
         for name, layer in model.named_children():
            if isinstance(layer,(LinearCRP,Conv2DCRP,MaxPool2dCRP,AdaptiveAvgPool2dCRP)):
                print(self.layer_length,layer)
                layer.register_forward_hook(self.hook_function)
                self.layer_length+=1
            else:
                if isinstance(layer,(nn.ReLU,nn.Flatten,nn.Identity,nn.BatchNorm2d,nn.MaxPool2d)):
                    continue
                else:
                    self.recursive_foward_hook(layer,self.layer_length)
            


    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.model(input)
    

    def interpet(self,input,output_type="max",rule="lrpepsilon",concept_ids={},parameter={},estimate_relevance={},concept_type="relmax",top_num=2,input_zbetaplus=False):
        self.layer_relevance=[]
        output=self.model(input)
        if output_type=="softmax":
            relevance = torch.softmax(output, dim=-1)
            dict_relevance={"":relevance}
        elif output_type=="max":
            predictions = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(predictions, dim=1, keepdim=True)
            mask = (predictions == max_values)
            relevance = predictions * mask
            relevance=(relevance>0).float()
            relevance[relevance==0]=-1
            dict_relevance={"":relevance}
        else:
            dict_relevance={"":output}
        self.layer_relevance.append(dict_relevance)
        if not parameter:
            parameter=self.lrp_default_parameters[rule]
        Nlayer=len(list(self.layer_inputs.items()))-1
        if "all" in estimate_relevance:
            estimate_concept_all_layer=True
            top_num=estimate_relevance["all"]
        estimate_concept_all_layer=False
        for index,(layer,layer_input) in enumerate(reversed(self.layer_inputs.items())):
            if isinstance(layer,LinearCRP) or isinstance(layer,Conv2DCRP) or isinstance(layer,AdaptiveAvgPool2dCRP)or isinstance(layer,MaxPool2dCRP):
                #print(layer,layer_input.shape,relevance.shape)
                if index==Nlayer and input_zbetaplus:
                    rule="lrpzbetalh"
                    parameter=self.lrp_default_parameters["lrpepsilon"]
                concept_index=(Nlayer-index)
                if  concept_index in concept_ids:
                    concepts=concept_ids[concept_index]
                else:
                    concepts=None
                if estimate_concept_all_layer:
                    dict_relevance=layer.interpet(layer_input,dict_relevance,concepts,concept_type,top_num,rule,parameter)
                else:
                    if concept_index in estimate_relevance:
                        conceptestimation_type=concept_type
                        top_num=estimate_relevance[concept_index]
                    else:
                        conceptestimation_type=None
                    dict_relevance=layer.interpet(layer_input,dict_relevance,concepts,conceptestimation_type,top_num,rule,parameter)
                print(layer,dict_relevance.keys())
                #self.layer_relevance.append(dict_relevance)
                #print(relevance.sum(),relevance.shape)
        return dict_relevance