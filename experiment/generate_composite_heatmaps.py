# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Generate heatmaps for different composites
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import os
import sys
from pathlib import Path
from zennit.image import imsave
import torch.nn.functional as F
from torchvision.utils import save_image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helper_functions import save_dict,load_dict,compute_lrp


if __name__=="__main__":
    index=1
    
    config={    
                "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
                "datasplitfilename": r"infos.csv",
                "n_classes":2,
                "filepath":r'D:/Thesis/ExplainingWilderness/experiments/wilderness_concepts/data/',
                "filename":r"wilderness_datasample.pkl",
                "datasaved":True,
                "device":"cuda",
                "datasettype":"test",
                "filterclass":[1],
                "models":["alexnet","vgg16","resnet18"],
                "modelweightpaths":[r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/alexnet_best_model.pth",
                                    r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/vgg16_best_model.pth",
                                    r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/resnet18_best_model.pth",
                                    ],
                "cmap":"bwr",
                "symmetric":True,
                "composite":["epsilon","epsilonplus","epsilonalphabeta","epsilonalphabetaflat","layerspecific"]
                }
    
    model=config["models"][index]
    modelweights=config["modelweightpaths"][index]
    datafilepath=r"./experiment/Notebooks/data/wilderness_data.pkl"
    loaded_data = load_dict( datafilepath)
    # Accessing the loaded images and labels
    images = loaded_data["images"][:-1,:,:,:].unsqueeze(dim=0).to(config["device"])
    images = loaded_data["images"].to(config["device"])
    
    labels = loaded_data["labels"]
    
    input_image_name="wilderness_data_composite.png"
    image_path = os.path.join(r"./experiment/Notebooks/temp/",input_image_name)
    save_image(F.pad(images, (2, 2, 2, 2), mode='constant', value=0), image_path, nrow=1, padding=0)
    for composite in config["composite"]:
        print("building heatmap composite type:",composite)
        heatmap,relevance=compute_lrp(model,config["n_classes"],modelweights,images,condition=[{"y":[0,1]}],compositname=composite,canonizerstype=model,outputtype="max")
        save_dict(relevance,os.path.join(r"./experiment/Notebooks/temp/relevance_dict",model+"_"+composite+"_relevance.pkl"))
        heatmap_path=os.path.join(r"./experiment/Notebooks/temp/heatmaps",model+"_"+composite+"_heatmap.png")
        imsave(heatmap_path,heatmap.to("cpu").numpy(),cmap=config["cmap"], symmetric=config["symmetric"], grid=(len(heatmap),1),level=1.2)
    
