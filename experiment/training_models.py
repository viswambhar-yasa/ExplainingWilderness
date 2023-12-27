# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Based on the hyperparameter obtained from tuning, we train the models using the config file
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: WanDB is used to track the progress 

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinelmodels.trainer import WildernessClassifier


# list of parameters the hyperparameter can select and perform the sweep operation, details given in chapter 8 
alexnetconfig={   
            "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
            "datasplitfilename": r"infos.csv",
            "n_classes":2,
            "device":"cuda",
            "inputimage_type":"rgb",    
            "modeltype":"alexnet",
            "modelweightspath":'',
            "modelweightspath":"",
            "trainable":True,
            "epochs": 25,
            "batchsize": 128,
            "lr": 0.001503,
            "optimizer":"sgd",
            "gamma":1.8,
            "lrscheduler":"step_lr",
            "losstype":"crossentropy",
            "lossweights":[0.8,1.6],
            "project_name":"wilderness-or-not",
            "log_images":True,
            "log_image_index":5,
            "earlystop_patience":3,
            "earlystop_min_delta":0.05,
            "classnames":["A","W"]
}


vgg16config={    
            "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
            "datasplitfilename": r"infos.csv",
            "n_classes":2,
            "device":"cpu",
            "inputimage_type":"rgb",   
            "modeltype":"vgg16",
            "modelweightspath":'',
            "trainable":True,
            "epochs": 2,
            "batchsize": 32,
            "lr": 0.001704,
            "optimizer":"adam",
            "gamma":1.5,
            "lrscheduler":"step_lr",
            "losstype":"crossentropy",
            "lossweights":[0.75,1.35],
            "project_name":"wilderness-or-not",
            "log_images":True,
            "log_image_index":5,
            "earlystop_patience":2,
            "earlystop_min_delta":0.05,
            "classnames":["A","W"]
}


resnet18config={
            "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
            "datasplitfilename": r"infos.csv",
            "n_classes":2,
            "device":"cuda",
            "inputimage_type":"rgb",   
            "modeltype":"resnet18",
            "modelweightspath":"",
            "trainable":True,
            "epochs": 25,
            "batchsize": 64,
            "lr": 0.001,
            "optimizer":"sgd",
            "gamma":1,
            "lrscheduler":"step_lr",
            "losstype":"crossentropy",
            "lossweights":[0.75,1.5],
            "project_name":"wilderness-or-not",
            "log_images":True,
            "log_image_index":5,
            "earlystop_patience":3,
            "earlystop_min_delta":0.05,
            "classnames":["A","W"]
}

#remove the pretrained_weights=None, for 'ImageNet' pretrained weight to be assigned to the model

config=alexnetconfig
csv_filepath=os.path.join(config["root_dir"],config["datasplitfilename"])
trainer=WildernessClassifier(csv_filepath, config["root_dir"],n_classes=config["n_classes"],input_imagetype=config["inputimage_type"],device=config["device"])
trainer.training(config,project_name=config["project_name"],log_images=config["log_images"],batch_idx=config["log_image_index"],patience=config["earlystop_patience"],
                 min_delta=config["earlystop_min_delta"],pretrained_weights=None)

