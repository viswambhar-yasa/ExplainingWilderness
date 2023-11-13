import torch 
import numpy as np
import matplotlib.pyplot as plt
from zennit.canonizers import CompositeCanonizer
from matplotlib.colors import LinearSegmentedColormap
from zennit.torchvision import VGGCanonizer,ResNetCanonizer
from zennit.composites import EpsilonPlus,EpsilonAlpha2Beta1,EpsilonPlusFlat,EpsilonAlpha2Beta1Flat
from interpet.concepts.conceptcomposite import EpsilonComposite,FlatComposite,EpsilonZboxComposite,EpsilonFlatComposite,CustomComposite


def canonizerstype(canonizersname):
    if canonizersname=="vgg":
        return [VGGCanonizer()]
    elif canonizersname=="epsilonalphabeta":
        return [ResNetCanonizer()]
    else:
        return [CompositeCanonizer()]

def compositetype(compositename,canonizername,**kwargs):
        canonizer=canonizerstype(canonizername)
        if compositename=="epsilonplus":
            return EpsilonPlus(canonizers=canonizer)
        elif compositename=="epsilonalphabeta":
            return EpsilonAlpha2Beta1(canonizers=canonizer)
        elif compositename=="epsilonplusflat":
            return EpsilonPlusFlat(canonizers=canonizer)
        elif compositename=="epsilonalphabetaflat":
            return EpsilonAlpha2Beta1Flat(canonizers=canonizer)
        elif compositename == "epsilon":
            return EpsilonComposite(canonizers=canonizer)
        elif compositename == "flat":
            return FlatComposite(**kwargs)
        elif compositename == "epsilonzbox":
            return EpsilonZboxComposite(**kwargs)
        elif compositename == "epsilonflat":
            return EpsilonFlatComposite(**kwargs)
        elif compositename == "customComposite":
            return CustomComposite(**kwargs)
        else:
            raise ValueError(f"Composite type '{compositename}' is not recognized.")
        
def get_relevance_function(output_type):
        def softmax_relevance(output):
            relevance = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(relevance, dim=1, keepdim=True)
            mask = (relevance == max_values)
            init_relevance = (-(mask == 0).float() + (mask > 0).float())
            return init_relevance

        def max_relevance(output):
            predictions = torch.softmax(output, dim=-1)
            max_values, _ = torch.max(predictions, dim=1, keepdim=True)
            mask = (predictions == max_values)
            relevance = predictions * mask
            init_relevance = (relevance > 0).float()
            # relevance[relevance==0]=-1
            return init_relevance

        def log_softmax_relevance(output):
            relevance = torch.softmax(output, dim=-1)
            init_relevance = torch.log(relevance / (1 - relevance))
            return init_relevance
        
        def max_activation(output):
            max_values, _ = torch.max(output, dim=1, keepdim=True)
            mask = (output == max_values)
            prediction_mask = output * mask
            classes=output.shape[-1]
            init_relevance=((-output*(prediction_mask==0).float())/(classes-1))+(output*(prediction_mask>0).float())
            return init_relevance

        # Define other relevance functions here
        if output_type == "softmax":
            return softmax_relevance
        elif output_type == "max":
            return max_relevance
        elif output_type == "log_softmax":
            return log_softmax_relevance
        elif output_type == "max_activation":
            return max_activation
        else:
            return None
        
def init_lrp(data,conditionattribute,compositename="epsilon",canonizerstype="vgg",output_type="log_softmax"):
        init_rel=get_relevance_function(output_type)
        composite=compositetype(compositename,canonizerstype)
        probabilties=torch.softmax(conditionattribute.model(data),dim=-1)
        conditions=[{"y":list(torch.unique(torch.argmax(probabilties, dim=-1)).squeeze().detach().numpy())}]
        heatmaps, _, relevance, _ =conditionattribute(data,conditions,composite,init_rel=init_rel)
        return heatmaps,relevance,probabilties


def plotcmap(colourlist=['#0055a4', '#ffffff', '#ef4135']):
        cmap = LinearSegmentedColormap.from_list('Custom', colourlist, N=1024)
        fig, ax = plt.subplots(figsize=(25, 1.5))
        gradient = np.linspace(0, 1, 1024)
        gradient = np.vstack((gradient, gradient))

        # Plot the color bar
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_title('Relevance Colour Map',y=1.5,va='center',fontsize=16)
        ax.set_axis_off()

        ax.text(-1, -1, 'negative relevance', ha='left', va='center', color='#0055a4', fontsize=12)
        ax.text(1024, -1, 'positive relevance', ha='right', va='center', color='#ef4135', fontsize=12)
        ax.text(538, -1, 'No relevance', ha='right', va='center', color='black', fontsize=12)
        plt.savefig('relevance_colurmap.png', dpi=300, bbox_inches='tight')
        pass