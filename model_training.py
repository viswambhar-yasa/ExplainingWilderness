import os
import torch
from wildernessmodel.preprocessing import SentinelDataset
from wildernessmodel.trainer import ANTHROPROTECTBinaryModel
from wildernessmodel.pretrained_model import PretrainedModel


def get_datasets(root_dir,csv_filepath,input_imagetype="rgb",n_classes=2,device="cpu"):
    """_summary_

    Args:
        root_dir (_type_): _description_
        csv_filepath (_type_): _description_
        input_imagetype (str, optional): _description_. Defaults to "rgb".
        n_classes (int, optional): _description_. Defaults to 2.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    train_dataset = SentinelDataset(csv_filepath, root_dir,input_imagetype,n_classes,None,datasettype="train",device=device)
    val_dataset = SentinelDataset(csv_filepath, root_dir,input_imagetype,n_classes,None,datasettype="val",device=device)
    test_dataset = SentinelDataset(csv_filepath, root_dir,input_imagetype,n_classes,None,datasettype="test",device=device)
    return train_dataset,val_dataset,test_dataset


if __name__=="__main__":
    print("Running training on wilderness data")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_per_process_memory_fraction(0.8)
    runtraining=True
    root_dir = 'D:/master-thesis/Dataset/anthroprotect'
    csvfilename="infos.csv"
    log_dir="./trainedmodels/log_vgg16bn_rgb_oc2_alllayers"
    #log_dir="./trainedmodels/log_alexnet_rgb_oc2_test"
    #log_dir="./trainedmodels/log_resnet_rgb_oc2_test"
    inputchannels=3
    output_channels=2
    trainbatch_size=8
    valbatch_size=16
    testbatch_size=16
    device="cpu"
    csv_filepath=os.path.join(root_dir,csvfilename)
    PM=PretrainedModel(inputchannels,output_channels)
    modelweigthpath=os.path.join(log_dir,"best_model.pth")
    if not os.path.exists(modelweigthpath):
        modelweigthpath=None
    #wildernessmodel,trainable_layer_names=PM.build_alexnet(modelweigthpath)
    wildernessmodel,trainable_layer_names=PM.build_vgg16bn(modelweigthpath)
    #wildernessmodel,trainable_layer_names=PM.build_resnet(modelweigthpath)
    print(wildernessmodel)
    for name, param in wildernessmodel.named_parameters():
        #if "classifier" or "fc" in name:
            param.requires_grad = True
        #else:
        #    param.requires_grad = False
    train_dataset,val_dataset,test_dataset=get_datasets(root_dir,csv_filepath)
    trainer=ANTHROPROTECTBinaryModel(wildernessmodel,train_dataset,val_dataset,test_dataset,output_channels,trainbatch_size,valbatch_size,testbatch_size,num_epochs=2,log_dir=log_dir,lr_step_size=1,lr_gamma=0.5,device="cpu")
    del wildernessmodel,PM,train_dataset,val_dataset,test_dataset
    if runtraining:
        trainer.train()
    trainer.evaluate()
