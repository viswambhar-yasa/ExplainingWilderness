import torch
import torch.nn as nn
from .modules.lrplinear import LinearLRP
from .modules.lrpconv2d import Conv2DLRP
from .modules.lrpadaptivepool import AdaptiveAvgPool2dLRP
from .modules.lrpmaxpool import MaxPool2dLRP

class LRPModel(nn.Module):
    
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model=model
        self.model.eval()
        self.converter(self.model)
        self.layer_inputs={}
        self.layer_relevance=[]
        self.recursive_foward_hook(self.model)

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
                setattr(model,name, LinearLRP(module))
            elif isinstance(module, nn.Conv2d):
                #print(name)
                setattr(model,name, Conv2DLRP(module))
            elif isinstance(module, nn.MaxPool2d):
                setattr(model,name, MaxPool2dLRP(module))
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                setattr(model,name, AdaptiveAvgPool2dLRP(module))
            else:
                self.converter(module)


    def hook_function(self,module,input,output):
        self.layer_inputs[module]=input[0]    

    def recursive_foward_hook(self,model):
         for name, layer in model.named_children():
            #if name=="downsampe":
            #    continue
            if isinstance(layer,(LinearLRP,Conv2DLRP,MaxPool2dLRP,AdaptiveAvgPool2dLRP)):
                layer.register_forward_hook(self.hook_function)
            else:
                if isinstance(layer,(nn.ReLU,nn.Flatten,nn.Identity,nn.BatchNorm2d)):
                    continue
                else:
                    self.recursive_foward_hook(layer)


    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.model(input)
    
    def interpet(self,input,output_type="max",rule="lrpzplus",parameter={},input_zbetaplus=True):
        self.layer_relevance=[]
        output=self.model(input)
        if output_type=="softmax":
            relevance = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(relevance, dim=1, keepdim=True)
            mask = (relevance == max_values)
            relevance*=(-(mask==0).float()+(mask>0).float())
        elif output_type=="max":
            predictions = torch.softmax(output, dim=-1)
            print(predictions)
            max_values, _ = torch.max(predictions, dim=1, keepdim=True)
            mask = (predictions == max_values)
            relevance = predictions * mask
            relevance=(relevance>0).float()
            #relevance[relevance==0]=-1
        elif output_type=="log_softmax":
            relevance = torch.softmax(output, dim=-1)
            relevance=torch.log(relevance / (1 - relevance))
        elif output_type=="softmax_grad":
            predictions = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(output, dim=-1, keepdim=True)
            mask = (output == max_values)
            prediction_mask = predictions * mask
            print(prediction_mask,(-(predictions*output)*(mask==0).float()))
            relevance=prediction_mask+((predictions*output)*(mask==0).float())
        elif output_type=="max_activation":
            max_values, _ = torch.max(output, dim=1, keepdim=True)
            mask = (output == max_values)
            prediction_mask = output * mask
            classes=output.shape[-1]
            relevance=((-output*(prediction_mask==0).float())/(classes-1))+(output*(prediction_mask>0).float())
        else:
            if (output.shape[1]==1):
                relevance=(output>0.5).float()
                relevance[relevance==0]=-1
            else:
                relevance=torch.zeros_like(output)
                relevance[:,0]=1
        print(relevance)
        self.layer_relevance.append(relevance)
        if not parameter:
            parameter=self.lrp_default_parameters[rule]
        Nlayer=len(list(self.layer_inputs.items()))-1
        for index,(layer,layer_input) in enumerate(reversed(self.layer_inputs.items())):
            if isinstance(layer,LinearLRP) or isinstance(layer,Conv2DLRP) or isinstance(layer,AdaptiveAvgPool2dLRP)or isinstance(layer,MaxPool2dLRP):
                #print(layer,layer_input.shape,relevance.shape)
                if index==Nlayer and input_zbetaplus:
                    rule="lrpzbetalh"
                    parameter=self.lrp_default_parameters["lrpepsilon"]
                relevance=layer.interpet(layer_input,relevance,rule,parameter)
                print(layer,relevance.shape)
                self.layer_relevance.append(relevance)
                #print(relevance.sum(),relevance.shape)
        return relevance
        #return relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach()
