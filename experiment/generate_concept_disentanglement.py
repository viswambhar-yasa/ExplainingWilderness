# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: We perform concept disentanglement, based on the layer and channel index
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: Graphs are created, which need to be handled properly or else it may create locking issue 
import os
import sys
from pathlib import Path
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpret.concept.conceptrelevance import ConceptRelevance
from sentinelmodels.pretrained_models import buildmodel
from helper_functions import load_dict,save_dict

config={    
                "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
                "datasplitfilename": r"infos.csv",
                "n_classes":2,
                "datasaved":True,
                "device":"cpu",
                "models":["alexnet","vgg16","resnet18"],
                "modelweightpaths":[r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/alexnet_best_model.pth",
                                    r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/vgg16_best_model.pth",
                                    r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/resnet18_best_model.pth",
                                    ],
                "cmap":"hot",
                "symmetric":False,
                "level":1
                }

modelindex=-1
datafilepath=r"D:/finaldraft/ExplainingWilderness/experiment/Notebooks/data/wilderness_data.pkl"

loaded_data = dictionary = load_dict(datafilepath)
# Accessing the loaded images and labels
images = loaded_data["images"][-1,:,:,:].unsqueeze(dim=0).to(config["device"])
#images = loaded_data["images"].to(config["device"])

labels = loaded_data["labels"]

input_image_name="wilderness_data_concept_disentanglement.png"
image_path = os.path.join(r"D:/finaldraft/ExplainingWilderness/experiment/Notebooks/temp/",input_image_name)
save_image(F.pad(images, (2, 2, 2, 2), mode='constant', value=0), image_path, nrow=1, padding=0)


model=buildmodel(model_type=config["models"][modelindex],multiclass_channels=config["n_classes"],modelweightpath=config["modelweightpaths"][modelindex]).to(config["device"])
Concepts=ConceptRelevance(model,device=config["device"])
del model

recordlayers=list(Concepts.layer_map.keys())
recordlayers=['features.0', 'features.4.0.conv1', 'features.4.0.conv2', 'features.4.1.conv1', 'features.4.1.conv2', 'features.5.0.conv1', 'features.5.0.conv2', 'features.5.1.conv1', 'features.5.1.conv2', 'features.6.0.conv1', 'features.6.0.conv2', 'features.6.1.conv1', 'features.6.1.conv2', 'features.7.0.conv1', 'features.7.0.conv2', 'features.7.1.conv1', 'features.7.1.conv2', 'common_layers.0', 'common_layers.3', 'common_layers.6']
recordlayers=['features.7.0.conv1', 'features.7.0.conv2', 'features.7.1.conv1']
layername='features.7.1.conv1'
initial_channel=485 # selecting channel
width=[3,2,1]
subconcepts=Concepts.compute_concept_disentangle(images,initial_channel,conceptlayer=layername,higher_concept_index=1,compositename="epsilonplus",canonizerstype=config["models"][modelindex],width=width,record_layers=recordlayers,build=True,abs_norm=True)

savepath=r"D:/finaldraft/ExplainingWilderness/experiment/Notebooks/temp/relevance_dict/disentangled_concepts"
save_dict(subconcepts,os.path.join(savepath,config["models"][modelindex]+"_"+layername+"conceptdisentanglement"+"_relevance.pkl"))
