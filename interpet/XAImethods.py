# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: A XAI class with many explainable methods like IntegratedGradients,GuidedBackprop,GradientShap,GuidedGradCam,Occlusion,Lime which state of the art methods
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: The lime method requires a feature mask as it is computational expensive.

import os
import copy
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients,GuidedBackprop,GradientShap,GuidedGradCam,Occlusion,Lime,Deconvolution,NoiseTunnel


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it
    takken from stack over flow https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class XAI:
    """
    The `XAI` class is a class that provides explainability methods for machine learning models. It allows users to interpret the predictions made by a given model by generating visual explanations for the model's decision-making process.

    Example Usage:
        # Create an instance of the XAI class
        xai = XAI(model)

        # Explain the features of a given input data using Integrated Gradients method
        features = xai.explain_features(data, explainmethod="Integrated Gradients")

        # Run all available explainability methods on the input data and save the visual explanations
        xai.run_all(data, methodlist=["Guided BackProp", "Grad SHAP", "Guided GradCam", "Occlusion", "Integrated Gradients"], path="./explanations", filename="explanation")

    Main functionalities:
    - Deep copy the input model and set it to evaluation mode
    - Initialize the explainability methods (Integrated Gradients, Guided Backprop, Grad SHAP, Guided GradCam, Occlusion, LIME)
    - Extract the first layer weights of the model
    - Get the predicted labels for a given input data
    - Explain the features of the input data using different explainability methods
    - Run all available explainability methods on the input data and save the visual explanations

    Methods:
    - __init__(self, model, device=None): Initializes the XAI class with a model and device (optional)
    - extract_first_layer_weights(self, model): Recursively finds and returns the first layer of the model
    - get_prediction(self, data): Returns the predicted labels for a given input data
    - integratedgradients(self, multiply_by_inputs): Initializes the Integrated Gradients method
    - guidedbackprop(self): Initializes the Guided Backprop method
    - gradshap(self): Initializes the Grad SHAP method
    - occlusion(self): Initializes the Occlusion method
    - limemethod(self): Initializes the LIME method
    - guidedgradcam(self, layer=None): Initializes the Guided GradCam method with an optional layer parameter
    - listofmethods(self): Returns a list of available explainability methods
    - explain_features(self, data, target=None, explainmethod="Integrated Gradients", multiply_by_inputs=True, eps=1e-10, baselinetype="zeros", n_steps=10, feature_mask=None, layer=None, interpolation="nearest", sliding_window_shapes=(1,3,3)): Explains the features of the input data using the specified explainability method
    - run_all(self, data, target=None, methodlist=None, multiply_by_inputs=True, eps=1e-10, baselinetype="zeros", n_steps=10, feature_mask=None, layer=None, interpolation="nearest", path="./", filename="", default_cmap="hot", method='heat_map', fig_size=(5,5), strides=(3,15,15), sliding_window_shapes=(3,30,30)): Runs all available explainability methods on the input data and saves the visual explanations

    Fields:
    - model: The input model
    - device: The device on which the model is being run
    - IG: The Integrated Gradients method
    - OC: The Occlusion method
    - DL: The DeepLift method
    - GB: The Guided Backprop method
    - Shap: The Grad SHAP method
    - CAM: The Guided GradCam method
    - lime: The LIME method
    """
    def __init__(self,model,device=None) -> None:
        
        self.model=copy.deepcopy(model)
        self.model.eval()
        self.model.zero_grad()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=torch.device(device)
        self.IG=None
        self.OC=None
        self.DL=None
        self.GB=None
        self.Shap=None
        self.CAM=None
        self.lime=None
        self.deconv=None
        self.nt=None
        pass
    
    def extract_first_layer_weights(self,model):
        """
        Recursive helper function to traverse nested structures and extract the first layer of a given model.

        Args:
            model (nn.Module): The input model for which the first layer needs to be extracted.

        Returns:
            nn.Module: The first layer of the given model. If no first layer is found, None is returned.
        """
        def find_first_layer(module):
            """
            Recursive function to find the first layer in a nested module.

            Args:
                module (nn.Module): The module to search for the first layer.

            Returns:
                nn.Module: The first layer found in the nested modules. If no first layer is found, None is returned.
            """
            for layer in module.children():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    return layer
                else:
                    # Recursively search in the nested modules
                    result = find_first_layer(layer)
                    if result is not None:
                        return result
            return None

      # Call the recursive helper function
        return find_first_layer(model)
    
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
    
    def guidedbackprop(self):
      self.GB=GuidedBackprop(self.model)

    def gradshap(self):
      self.Shap=GradientShap(self.model)

    def occlusion(self):
      self.OC=Occlusion(self.model)
    
    def limemethod(self):
      self.lime=Lime(self.model)
    
    def decon(self):
        self.deconv = Deconvolution(self.model)
    
    def guidedgradcam(self,layer=None):
      if layer is None:
        layer_name=self.extract_first_layer_weights(self.model)
      self.CAM=GuidedGradCam(self.model,layer=layer_name)
    
    def noisetunnel(self,layer=None):
      self.nt=NoiseTunnel(self.IG)

    def listofmethods(self):
        methods=["Intergrated Gradients","NoiseTunnel","Guided BackProp","Grad SHAP","Guided GradCam","Deconvolution","Occlusion"]
        return methods

    def explain_features(self, data, target=None, explainmethod="Integrated Gradients", multiply_by_inputs=True, eps=1e-10, baselinetype="zeros", n_steps=10, feature_mask=None, layer=None, interpolation="nearest", sliding_window_shapes=(1,3,3)):
        """
        Compute the feature attributions for the input data using different explainability methods.

        Args:
            data (torch.Tensor): The input data for which the features need to be explained.
            target (torch.Tensor, optional): The target labels for the input data. If not provided, the method will use the predicted labels.
            explainmethod (str, optional): The explainability method to be used. Default is "Integrated Gradients".
            multiply_by_inputs (bool, optional): Whether to multiply the gradients by the inputs. Default is True.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-10.
            baselinetype (str, optional): The type of baseline to be used. Default is "zeros".
            n_steps (int, optional): The number of steps for the explainability method. Default is 10.
            feature_mask (torch.Tensor, optional): A mask to specify which features to explain. Default is None.
            layer (str, optional): The layer to be used for the Guided GradCam method. Default is None.
            interpolation (str, optional): The interpolation mode for the Guided GradCam method. Default is "nearest".
            sliding_window_shapes (tuple, optional): The shape of the sliding window for the Occlusion method. Default is (1,3,3).

        Returns:
            torch.Tensor: The computed feature attributions for the input data.
        """
        if not data.requires_grad:
            data.requires_grad = True

        if target is None:
            targetlabels = self.get_predicition(data)
        else:
            targetlabels = target

        if baselinetype == "zeros":
            baseline = data * 0
        else:
            baseline = torch.randn_like(data)

        if explainmethod == "Guided BackProp":
            if self.GB is None:
                self.guidedbackprop()
            features = self.GB.attribute(inputs=data, target=targetlabels)
        elif explainmethod == "Grad SHAP":
            if self.Shap is None:
                self.gradshap()
            features = self.Shap.attribute(inputs=data, target=targetlabels, n_samples=n_steps, feature_mask=feature_mask)
        elif explainmethod == "Guided GradCam":
            if self.CAM is None:
                self.guidedgradcam(layer)
            features = self.CAM.attribute(inputs=data, target=targetlabels, interpolate_mode=interpolation)
        elif explainmethod == "Occlusion":
            if self.OC is None:
                self.occlusion()
            features = self.OC.attribute(inputs=data, target=targetlabels, sliding_window_shapes=sliding_window_shapes)
        elif explainmethod == "Deconvolution":
            if self.lime is None:
                self.decon()
            features = self.deconv.attribute(inputs=data, baselines=baseline, feature_mask=feature_mask, target=targetlabels, n_samples=n_steps)
        elif explainmethod == "LIME":
            if self.lime is None:
                self.limemethod()
            features = self.lime.attribute(inputs=data, baselines=baseline, feature_mask=feature_mask, target=targetlabels, n_samples=n_steps)
        elif explainmethod=="NoiseTunnel":
                if self.IG is None:
                    self.integratedgradients(multiply_by_inputs)
                if self.nt is None:
                    self.noisetunnel()
                features = self.nt.attribute(inputs=data,nt_type='smoothgrad',nt_samples=10, target=targetlabels)
        else:
            if self.IG is None:
                self.integratedgradients(multiply_by_inputs)
            features = self.IG.attribute(inputs=data, target=targetlabels, baselines=baseline, n_steps=n_steps)

        return features

    def run_all(self, data, target=None, methodlist=None, multiply_by_inputs=True, eps=1e-10, baselinetype="zeros", n_steps=10, feature_mask=None, layer=None, interpolation="nearest", path="./", filename="", default_cmap="hot", method='heat_map', fig_size=(5,5), strides=(3,15,15), sliding_window_shapes=(3,30,30)):
        """
        Runs different explainability methods on the input data and saves the visual explanations of the features using the specified methods.

        Args:
            data (torch.Tensor): The input data for which the features need to be explained.
            target (torch.Tensor, optional): The target labels for the input data. If not provided, the method will use the predicted labels.
            methodlist (list, optional): A list of explainability methods to be used. If not provided, all available methods will be used.
            multiply_by_inputs (bool, optional): Whether to multiply the gradients by the inputs. Default is True.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-10.
            baselinetype (str, optional): The type of baseline to be used. Default is "zeros".
            n_steps (int, optional): The number of steps for the explainability method. Default is 10.
            feature_mask (torch.Tensor, optional): A mask to specify which features to explain. Default is None.
            layer (str, optional): The layer to be used for the Guided GradCam method. Default is None.
            interpolation (str, optional): The interpolation mode for the Guided GradCam method. Default is "nearest".
            path (str, optional): The path to save the visual explanations. Default is "./".
            filename (str, optional): The base filename for the saved visual explanations. Default is "".
            default_cmap (str, optional): The default colormap for the visual explanations. Default is "hot".
            method (str, optional): The visualization method for the visual explanations. Default is "heat_map".
            fig_size (tuple, optional): The size of the figure for the visual explanations. Default is (5,5).
            strides (tuple, optional): The strides for the sliding window in the Occlusion method. Default is (3,15,15).
            sliding_window_shapes (tuple, optional): The shape of the sliding window in the Occlusion method. Default is (3,30,30).

        Returns:
            dict: A dictionary containing the feature attributions and visual explanations for each explainability method. The keys are the explainability method names, and the values are tuples containing the feature attributions and the visual explanations.
        """
        if not data.requires_grad:
            data.requires_grad = True
        if target is None:
            targetlabels = self.get_predicition(data)
        else:
            targetlabels = target
        if baselinetype == "zeros":
            baseline = data * 0
        else:
            baseline = torch.randn_like(data)
        if methodlist is None:
            methodlist = self.listofmethods()
        explainationdict = {}
        plt.ioff()
        for explainmethod in methodlist:
            if explainmethod == "Guided Backprop":
                if self.GB is None:
                    self.guidedbackprop()
                feature = self.GB.attribute(inputs=data, target=targetlabels)
            elif explainmethod == "Grad SHAP":
                if self.Shap is None:
                    self.gradshap()
                feature = self.Shap.attribute(inputs=data, baselines=baseline, target=targetlabels, n_samples=n_steps)
            elif explainmethod == "Guided GradCam":
                if self.CAM is None:
                    self.guidedgradcam(layer)
                feature = self.CAM.attribute(inputs=data, target=targetlabels, interpolate_mode=interpolation)
            elif explainmethod == "Occlusion":
                if self.OC is None:
                    self.occlusion()
                feature = self.OC.attribute(inputs=data, target=targetlabels, strides=strides, sliding_window_shapes=sliding_window_shapes)
            elif explainmethod == "LIME":
                if self.lime is None:
                    self.limemethod()
                feature = self.lime.attribute(inputs=data, baselines=baseline, feature_mask=feature_mask, target=targetlabels, n_samples=n_steps)
            elif explainmethod == "Deconvolution":
                if self.lime is None:
                    self.decon()
                
            elif explainmethod=="NoiseTunnel":
                if self.IG is None:
                    self.integratedgradients(multiply_by_inputs)
                if self.nt is None:
                    self.noisetunnel()
                features = self.nt.attribute(inputs=data,nt_type='smoothgrad',nt_samples=10, target=targetlabels)
            else:
                if self.IG is None:
                    self.integratedgradients(multiply_by_inputs)
                feature = self.IG.attribute(inputs=data, target=targetlabels, baselines=baseline, n_steps=n_steps)
            heatmap= viz.visualize_image_attr(np.transpose(feature.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                  np.transpose(data.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                  method=method,
                                                  cmap=default_cmap,
                                                  show_colorbar=False,
                                                  sign='positive',
                                                  outlier_perc=1,
                                                  fig_size=fig_size,use_pyplot=False,

                                                  )
            maskheatmap= viz.visualize_image_attr(np.transpose(feature.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                  np.transpose(data.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                  method="masked_image",
                                                  cmap=default_cmap,
                                                  show_colorbar=False,
                                                  sign='positive',
                                                  outlier_perc=1,
                                                  fig_size=fig_size,use_pyplot=False,

                                                  )
            heatmap[0].tight_layout()
            file_name = filename + "_" + explainmethod
            filepath = os.path.join(path, file_name + ".png")  # Specify the filename and format
            heatmap[0].savefig(filepath, dpi=600, bbox_inches='tight', format="png", pad_inches=0)
            maskheatmap[0].tight_layout()
            maskfilepath = os.path.join(path, file_name + "_mask.png") 
            maskheatmap[0].savefig(maskfilepath, dpi=600, bbox_inches='tight', format="png", pad_inches=0)
            explainationdict[explainmethod] = (feature, Image.open(filepath),Image.open(maskfilepath))
            
            plt.close(heatmap[0])
            
        return explainationdict
