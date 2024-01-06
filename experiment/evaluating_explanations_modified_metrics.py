
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
from interpret.XAIevaluation import XAIEvaluation



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
                "subset_size":256,
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
            "subset_size":256,
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
            "subset_size":256,
            "classnames":["Anthropogenic","wilderness"],
            "project_name":"wilderness-or-not"
}



import quantus
import numpy as np
xai=XAIEvaluation(resnet18config)
metrics = {
                "Robustness": quantus.AvgSensitivity(
                    nr_samples=10,
                    lower_bound=0.1,
                    norm_numerator=quantus.norm_func.fro_norm,
                    norm_denominator=quantus.norm_func.fro_norm,
                    perturb_func=quantus.perturb_func.uniform_noise,
                    similarity_func=quantus.similarity_func.difference,
                    abs=False,
                    normalise=True,
                    disable_warnings=True,
                ),
                "Faithfulness": quantus.FaithfulnessCorrelation(
                    nr_runs=200,
                    subset_size=256,
                    perturb_baseline="black",
                    abs=False,
                    normalise=True,
                    disable_warnings=True,
                ),
                "Complexity": quantus.Sparseness(
                    disable_warnings=True,
                )
                }

score=xai.runevaluation(xaimethodslist=["LRP","CRP"],metricsdict=metrics,stopstep=4)
saved_path=r"D:/finaldraft/ExplainingWilderness/experiment/Notebooks/temp"
filename="_evaluation_metric_dict_optimised1.pkl"
save_dict(score,os.path.join(saved_path,resnet18config["modeltype"]+filename))
print(score)