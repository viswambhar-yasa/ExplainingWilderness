# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Perform hyperparameter tuning for the selected model
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: WanDB is used to perform the hyperparameter tuning and sweeps are performed by the agents

import os
import sys
import wandb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinelmodels.trainer import WildernessClassifier

root_dir = r'D:/master-thesis/Dataset/anthroprotect'
csvfilename=r"infos.csv"
csv_filepath=os.path.join(root_dir,csvfilename)
trainer=WildernessClassifier(csv_filepath, root_dir,input_imagetype="rgb",device="cuda")

sweep_config = {'method': 'random',
            'metric':{'name': 'loss','goal': 'minimize'}}

# list of parameters the hyperparameter can select and perform the sweep operation, details given in chapter 8 
parameters_dict = {
        'modeltype':{'value':"resnet18"},
        'modelweightspath':{'value':''},
        'optimizer': {'values': ['adam', 'sgd']},
        "lrscheduler":{'values':["step_lr","exponential_lr"]},
        'lr': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1,
          },
        'batchsize': {
            'distribution': 'q_log_uniform_values',
            'q': 4,
            'min': 4,
            'max': 16,},
        "losstype":{"value":"crossentropy"},
        "gamma":{
            'distribution': 'uniform',
            'min': 0,
            'max': 2,
        },
        "seed":{"value":69},
        'epochs': {
            'value': 1},
        'subsetsize':{'value':0.15},
        'lossweights':{'values':[[1,1],[0.75,1.25]]}
            }

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project=parameters_dict["modeltype"]["value"]+"_pretrained_model")

wandb.agent(sweep_id, trainer.hyperparameter_tuning, count=20,project=parameters_dict["modeltype"]["value"]+"_pretrained_model")

