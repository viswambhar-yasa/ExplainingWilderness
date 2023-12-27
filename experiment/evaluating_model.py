# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: This file perform evaluation of the model
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: WanDB is used to track the progress 

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinelmodels.trainer import WildernessClassifier

config={   
            "root_dir":r'D:/master-thesis/Dataset/anthroprotect',
            "datasplitfilename": r"infos.csv",
            "n_classes":2,
            "device":"cuda",
            "inputimage_type":"rgb",    
            "modeltype":"resnet18",
            "modelweightspath":r'D:/Thesis/ExplainingWilderness/experiments/figures/trained_models/pretrained/resnet18_best_model.pth',
            "batchsize": 16,
            "classnames":["Anthropogenic","wilderness"],
            "project_name":"wilderness-or-not"
}



csv_filepath=os.path.join(config["root_dir"],config["datasplitfilename"])
trainer=WildernessClassifier(csv_filepath, config["root_dir"],n_classes=config["n_classes"],input_imagetype=config["inputimage_type"],device=config["device"])
trainer.evaluate(config=config,project_name=config["project_name"])
