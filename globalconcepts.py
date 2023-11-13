import os
import torch
from interpet.concepts.conceptpropagation import ConceptRelevance
#from interpet.concepts.conceptcomposite import get
from wildernessmodel.preprocessing import SentinelDataset
from wildernessmodel.pretrained_model import PretrainedModel



def generate_globalconcept(dataset,model,batchsize,preprocessing,filesavepath,build=True,compositename="epsilonplusflat",canonizerstype="vgg",device="cpu",max_target="max",chkpoint=250,relevance_range=(0,8),receptivefield=False):
    Concepts=ConceptRelevance(model,device=device)
    Concepts.compute_concepts(dataset,preprocessing,filesavepath,compositename=compositename,canonizerstype=canonizerstype,device=device,imagecache=True,imagecachefilepath="cache",max_target=max_target,build=build,batch_size=batchsize,chkpoint=chkpoint)
    Concepts.glocal_analysis(compositename,canonizerstype,relevance_range,receptivefield,batchsize=16)
    pass


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
    batch_size=8
    device="cpu"
    csv_filepath=os.path.join(root_dir,csvfilename)
    PM=PretrainedModel(inputchannels,output_channels)
    modelweigthpath=os.path.join(log_dir,"best_model.pth")
    if not os.path.exists(modelweigthpath):
        modelweigthpath=None
    #wildernessmodel,trainable_layer_names=PM.build_alexnet(modelweigthpath)
    wildernessmodel,trainable_layer_names=PM.build_vgg16bn(modelweigthpath)
    val_dataset = SentinelDataset(csv_filepath, root_dir,datasettype="val",device=device)
    filesavepath="conceptdata/wilderness"
    build=True,
    compositename="epsilonplusflat"
    canonizerstype="vgg"
    device="cpu"
    max_target="max"
    chkpoint=50
    relevance_range=(0,8)
    receptivefield=False
    Concepts=ConceptRelevance(wildernessmodel,device=device)
    Concepts.compute_concepts(val_dataset,preprocessing=lambda x: x,filesavepath=filesavepath,compositename=compositename,canonizerstype=canonizerstype,device=device,imagecache=True,imagecachefilepath="cache",max_target=max_target,build=build,batch_size=batch_size,chkpoint=chkpoint)
    #Concepts.glocal_analysis(compositename,canonizerstype,relevance_range,receptivefield,batchsize=8)
    