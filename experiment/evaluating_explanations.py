
# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Performs evalaution by calling XAIEvaluation class and run all explainable methods for their corresponding metrics
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 


import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment.helper_functions import save_dict
from interpet.XAIevaluation import XAIEvaluation



alexnetconfig={    
                "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
                "datasplitfilename": r"infos.csv",
                "n_classes":2,
                "device":"cuda",
                "datasettype":"test",
                "filterclass":None,
                "inputimage_type":"rgb", 
                "modeltype":"alexnet",
                "modelweightspath":r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/alexnet_best_model.pth",                 
                "batchsize": 16,
                "subset_size":16,
                }



resnet18config={   
            "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
            "datasplitfilename": r"infos.csv",
            "n_classes":2,
            "device":"cuda",
            "datasettype":"test",
            "filterclass":None,
            "inputimage_type":"rgb",    
            "modeltype":"resnet18",
            "modelweightspath":r'D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/resnet18_best_model.pth',
            "batchsize": 8,
            "subset_size":16,
            "classnames":["Anthropogenic","wilderness"],
            "project_name":"wilderness-or-not"
}



vgg16config={   
            "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
            "datasplitfilename": r"infos.csv",
            "n_classes":2,
            "device":"cuda",
            "datasettype":"test",
            "filterclass":None,
            "inputimage_type":"rgb",    
            "modeltype":"vgg16",
            "modelweightspath":r"D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/vgg16_best_model.pth",
            "batchsize": 4,
            "subset_size":16,
            "classnames":["Anthropogenic","wilderness"],
            "project_name":"wilderness-or-not"
}




xai=XAIEvaluation(resnet18config)
score=xai.runevaluation(xaimethodslist=["IntergratedGradients","GradientShap","GuidedGradCam","LRP","CRP","Occulsion"])
saved_path=r"D:/finaldraft/ExplainingWilderness/experiment/Notebooks/temp"
filename="_evaluation_metric_dict_1.pkl"
save_dict(score,os.path.join(saved_path,resnet18config["modeltype"]+filename))
print(score)