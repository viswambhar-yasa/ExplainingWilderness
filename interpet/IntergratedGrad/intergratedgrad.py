import torch
import torch.nn as nn
from copy import deepcopy

class IntergratedGrad(nn.Module):
    def __init__(self,model:torch.nn.Module) -> None:
        super().__init__()
        self.model=deepcopy(model)
        self.model.eval()
        self.layer_grads={}
        pass

    def backward_hook_func(self,module, input_grad,ouptut_grad):
        self.layer_grads[module]=input_grad[0]

    def register_backward_hook(self,model):
         for name, layer in model.named_children():
            if isinstance(layer,(nn.ReLU,nn.Flatten,nn.Identity,nn.BatchNorm2d,nn.Linear,nn.Conv2d,nn.AdaptiveAvgPool2d,nn.Dropout)):
                layer.register_forward_hook(self.hook_function)
            else:
                self.register_backward_hook(layer)

    def integratedGradients(self,inputs,predictedclass,baseinput=None,totalsteps=50,step_size=1):
        inputs.requires_grad_(True)
        if baseinput is None:
            baseinput=torch.zeros_like(inputs)
        attributes=0
        for step in  range(1,totalsteps):
            Riemmann_approximation=baseinput+(step/totalsteps)*(inputs-baseinput)
            output=self.model(Riemmann_approximation)[:,int(predictedclass)]
            output.backward()
            attributes+=inputs.grad[0]/totalsteps
            #inputs.grad.zero()
        return attributes
    
    def forward(self,inputs):
        output=self.model(inputs)
        return output
    
    def interpet(self,inputs,output_type="softmax",baseinput=None,totalsteps=50,step_size=1):
        output=self.model(inputs)
        if output_type=="softmax":
            predictions = torch.softmax(output, dim=-1)
            predictedclass, _ = torch.max(predictions, dim=1, keepdim=True)
        elif output_type=="log_softmax":
            predictions = torch.softmax(output, dim=-1)
            log_predictions=torch.log(predictions / (1 - predictions))
            predictedclass, _ = torch.max(log_predictions, dim=1, keepdim=True)
        attributes=self.integratedGradients(inputs,predictedclass,baseinput,totalsteps,step_size)
        IntergratedGrad_attributes=attributes*inputs
        return IntergratedGrad_attributes,attributes

        





    