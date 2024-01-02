# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: This file perform generated explaination for other state of the art methods
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: WanDB is used to track the progress 

import os
import sys
from pathlib import Path
import torch.nn.functional as F
from torchvision.utils import save_image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from interpret.XAImethods import XAI
from experiment.helper_functions import load_dict
from sentinelmodels.pretrained_models import buildmodel


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
                "plotmethod":"heat_map"
                }

datafilepath=r"D:/finaldraft/ExplainingWilderness/experiment/Notebooks/data/wilderness_data.pkl"
saved_path="D:/finaldraft/ExplainingWilderness/experiment/Notebooks/temp/heatmaps/"
modelindex=-1
imagename=config["models"][modelindex]

loaded_data = dictionary = load_dict(datafilepath)
# Accessing the loaded images and labels
images = loaded_data["images"][-1,:,:,:].unsqueeze(dim=0).to(config["device"])
#images = loaded_data["images"].to(config["device"])

labels = loaded_data["labels"]

input_image_name="wilderness_data_XAI.png"
image_path = os.path.join(r"D:/finaldraft/ExplainingWilderness/experiment/Notebooks/temp/",input_image_name)
save_image(F.pad(images, (2, 2, 2, 2), mode='constant', value=0), image_path, nrow=1, padding=0)

model=buildmodel(model_type=config["models"][modelindex],multiclass_channels=config["n_classes"],modelweightpath=config["modelweightpaths"][modelindex]).to(config["device"])

XAImethods=XAI(model,device=config['device'])
del model

XAImethodslist=XAImethods.listofmethods()

print("XAI methods",XAImethodslist)

explaination=XAImethods.run_all(images,target=[1],methodlist=None,path=saved_path, filename=imagename, default_cmap=config["cmap"], method=config['plotmethod'],strides=(3,5,5),sliding_window_shapes=(3,10,10))