# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Based on the hyperparameter obtained from tuning, we train the models for multi label classification using the config file
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: WanDB is used to track the progress 

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinelmodels.trainer import WildernessMultlabelClassifier

resnet18config={
            "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
            "datasplitfilename": r"infos.csv",
            "n_classes":2,
            "n_labels":10,
            "device":"cuda",
            "inputimage_type":"rgb",   
            "modeltype":"vgg16",
            #"modelweightspath":r"D:/finaldraft/ExplainingWilderness/wandb/run-20240104_183637-d6tzchdv/files/alexnet\best_model.pth",
            "modelweightspath":"",
            "trainable":False,
            "epochs": 5,
            "batchsize": 8,
            "lr": 0.001,
            "optimizer":"adam",
            "gamma":1.5,
            "lrscheduler":"step_lr",
            "losstype":"binaryloss",
            "lossweights":list(np.ones(10)),
            "project_name":"wildernesslabels",
            "log_images":True,
            "log_image_index":5,
            "earlystop_patience":5,
            "earlystop_min_delta":0.05,
            "classnames":["A","W"]
}


config=resnet18config

labels_count = np.array([15934., 12054., 22., 1857., 1231., 16766., 10498., 2479., 5506., 6556.])

total_samples = np.sum(labels_count)
class_weights = total_samples / (len(labels_count) * labels_count)

print("Class Weights:", class_weights)
config["lossweights"]=class_weights
csv_filepath=os.path.join(config["root_dir"],config["datasplitfilename"])
broad_categories =  ["Artificial Surfaces","Arable land","Permanent crops","Pastures","heterogeneous agricultural areas", "Forest","scrub ","open spaces", "Wetlands", "Waterbodies"]       
trainer=WildernessMultlabelClassifier(csv_filepath, config["root_dir"],n_classes=config["n_classes"],nlabels=config['n_labels'],channel_name="corine2",input_imagetype=config["inputimage_type"],device=config["device"])
config["classnames"]=broad_categories
trainer.training(config,project_name=config["project_name"],log_images=config["log_images"],batch_idx=config["log_image_index"],patience=config["earlystop_patience"],
                 min_delta=config["earlystop_min_delta"],pretrained_weights=None)

