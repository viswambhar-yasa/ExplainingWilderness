# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Generate heatmaps for different models
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import os
import sys
from pathlib import Path
from zennit.image import imsave
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.helper_functions import save_dict,load_dict,compute_lrp


if __name__=="__main__":
    config={    
                "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
                "datasplitfilename": r"infos.csv",
                "n_classes":2,
                "datasaved":True,
                "device":"cuda",
                "models":["alexnet","vgg16","resnet18"],
                "modelweightpaths":[r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/alexnet_best_model.pth",
                                    r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/vgg16_best_model.pth",
                                    r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/resnet18_best_model.pth",
                                    ],
                "cmap":"hot",
                "symmetric":False,
                }
    
    
    datafilepath=r"D:/Thesis/ExplainingWilderness/experiment/Notebooks/data/wilderness_data.pkl"
    
    loaded_data = load_dict( datafilepath)
    # Accessing the loaded images and labels
    images = loaded_data["images"][:-1,:,:,:].unsqueeze(dim=0).to(config["device"])
    images = loaded_data["images"].to(config["device"])
    
    labels = loaded_data["labels"]
    
    input_image_name="wilderness_data.png"
    image_path = os.path.join(r"./experiment//Notebooks/temp/",input_image_name)

    save_image(F.pad(images, (2, 2, 2, 2), mode='constant', value=0), image_path, nrow=1, padding=0)
    for model,modelweights in zip(config["models"],config["modelweightpaths"]):
        print("building heatmap :",model)
        heatmap,relevance=compute_lrp(model,config["n_classes"],modelweights,images,compositname="epsilonplus",canonizerstype=model)
        save_dict(relevance,os.path.join(r"./experiment//Notebooks/temp/relevance_dict",model+"_relevance_dict.pkl"))
        heatmapname=model+"_heatmap.png"
        heatmap_path=os.path.join(r"./experiment/Notebooks/temp/heatmaps",heatmapname)
        imsave(heatmap_path,heatmap.to("cpu").numpy(),cmap=config["cmap"], symmetric=config["symmetric"], grid=(len(heatmap),1),level=1)

    
