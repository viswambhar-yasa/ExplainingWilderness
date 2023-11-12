import torch
import torch.nn as nn
import copy

from captum.attr import IntegratedGradients,GuidedBackprop,DeepLift,ShapleyValueSampling,GuidedGradCam
class XAI:
    def __init__(self,model,device=None) -> None:
        """_summary_
  
        Args:
            model (_type_): _description_
            device (_type_, optional): _description_. Defaults to None.
        """
        self.model=copy.deepcopy(model)
        self.model.eval()
        self.model.zero_grad()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=torch.device(device)
        self.IG=None
        self.DL=None
        self.GB=None
        self.Shap=None
        self.CAM=None
        pass
    
    def extract_first_layer_weights(self):
      # Find the first convolutional or linear layer in the model
      first_layer = None
      for layer in self.model.children():
          if isinstance(layer, (nn.Conv2d, nn.Linear)):
              first_layer = layer
              break
      return first_layer
    
    def get_predicition(self,data):
      """_summary_

      Args:
          data (torch.Tensor): input data

      Returns:
         targetlabels : predicted 
      """
      output=self.model(data)
      targetlabels=torch.argmax(torch.softmax(output,dim=-1),dim=-1)
      return targetlabels
          
    def integratedgradients(self,multiply_by_inputs):
      self.IG=IntegratedGradients(self.model,multiply_by_inputs)
    
    def deeplift(self,multiply_by_inputs,eps):
      self.DL=DeepLift(self.model, multiply_by_inputs,eps)

    def guidedbackprop(self):
      self.GB=GuidedBackprop(self.model)

    def shapleyvaluesampling(self):
      self.Shap=ShapleyValueSampling(self.model)

    def guidedgradcam(self,layer=None):
      if layer is None:
        layer_name=self.extract_first_layer_weights()
      self.CAM=GuidedGradCam(self.model,layer=layer_name)

    def explain_features(self,data,target=None,explainmethod="integratedgradients",multiply_by_inputs=True,eps=1e-10,baselinetype="zeros",n_steps=10,feature_mask=None,layer=None,interpolation="nearest"): 
      if not data.requires_grad:
          data.requires_grad=True
      if target is None:
        targetlabels=self.get_predicition(data)
      else:
        targetlabels=target
      if baselinetype=="zeros":
           baseline=data*0
      
      if explainmethod=="deeplift":
        if self.DL is None:
          self.deeplift(multiply_by_inputs,eps)
        features=self.DL.attribute(inputs=data,target=targetlabels,baselines=baseline)
      elif explainmethod=="guidedbackprop":
        if self.GB is None:
          self.guidedbackprop()
        features=self.GB.attribute(inputs=data,target=targetlabels)
      elif explainmethod=="shapleyvaluesampling":
        if self.Shap is None:
          self.shapleyvaluesampling()
        features=self.Shap.attribute(inputs=data,target=targetlabels,n_samples=n_steps,feature_mask=feature_mask)
      elif explainmethod=="guidedgradcam":
        if self.CAM is None:
          self.guidedgradcam(layer)
        features=self.CAM.attribute(inputs=data,target=targetlabels,interpolate_mode=interpolation)
      else:
        if self.IG is None:
          self.integratedgradients(multiply_by_inputs)
        features=self.IG.attribute(inputs=data,target=targetlabels,baselines=baseline,n_steps=n_steps)
      return features
  