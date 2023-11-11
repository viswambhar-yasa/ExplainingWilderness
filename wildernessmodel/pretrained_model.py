import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedModel:
    def __init__(self,input_channels,output_channels,pretrained_weights="IMAGENET1K_V1") -> None:
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.pretrained_weights=pretrained_weights
        pass

    def build_vgg16bn(self,modelweightpath=None):
        pretrained_model= models.vgg16_bn(weights=self.pretrained_weights)
        num_features = pretrained_model.classifier[6].in_features
        # Define additional layers for the extended classifier
        additional_layer1 = nn.Linear(num_features, 1000)  # You can adjust the size of the hidden layer(s)
        additional_layer2 = nn.ReLU(inplace=True)
        additional_layer3 = nn.Dropout(0.5)  # Add dropout for regularization if needed
        additional_layer4 = nn.Linear(1000, 256)  # You can adjust the size of the hidden layer(s)
        additional_layer5 = nn.ReLU(inplace=True)
        additional_layer6 = nn.Dropout(0.5) 
        additional_layer7 = nn.Linear(256, self.output_channels)

        # Insert the additional layers into the existing classifier
        pretrained_model.classifier = nn.Sequential(
            *(list(pretrained_model.classifier.children())[:-1]) + [additional_layer1, additional_layer2, additional_layer3, additional_layer4,additional_layer5,additional_layer6,additional_layer7]
        )
        trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in pretrained_model.named_parameters() if param.requires_grad])))

        if modelweightpath is not None:
            pretrained_model.load_state_dict(torch.load(modelweightpath))

        return pretrained_model,trainable_layer_names
    
    def build_resnet(self,modelweightpath=None):
        pretrained_model=models.resnet18(weights=self.pretrained_weights)
        # Define the additional layers
        additional_layer1 = nn.Linear(pretrained_model.fc.in_features, 256)
        additional_layer2 = nn.ReLU(inplace=True)
        additional_layer3 = nn.Dropout(0.5)
        additional_layer4 = nn.Linear(256, 256)
        additional_layer5 = nn.ReLU(inplace=True)
        additional_layer6 = nn.Dropout(0.5)
        additional_layer7 = nn.Linear(256, self.output_channels)  # Assuming 10 output classes, adjust as needed

        # Modify the model by replacing the original fully connected layer with the additional layers
        pretrained_model.fc = nn.Sequential(
            additional_layer1,
            additional_layer2,
            additional_layer3,
            additional_layer4,
            additional_layer5,
            additional_layer6,
            additional_layer7
        )
        trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in pretrained_model.named_parameters() if param.requires_grad])))

        if modelweightpath is not None:
            pretrained_model.load_state_dict(torch.load(modelweightpath))
        return pretrained_model,trainable_layer_names
    
    def build_alexnet(self,modelweightpath=None):
        pretrained_model=models.alexnet(weights=self.pretrained_weights)
        # Define the additional layers
        additional_layer1 = nn.Linear(4096, 2048)
        additional_layer2 = nn.ReLU(inplace=True)
        additional_layer3 = nn.Dropout(0.5)
        additional_layer4 = nn.Linear(2048, 1024)
        additional_layer5 = nn.ReLU(inplace=True)
        additional_layer6 = nn.Dropout(0.5)
        additional_layer7 = nn.Linear(1024, 256)
        additional_layer8 = nn.ReLU(inplace=True)
        additional_layer9 = nn.Dropout(0.5)
        additional_layer10 = nn.Linear(256, self.output_channels)  # Assuming 10 output classes, adjust as needed

        # Modify the model by replacing the original fully connected layer with the additional layers
        pretrained_model.classifier = nn.Sequential(
            pretrained_model.classifier[0],
            pretrained_model.classifier[1],
            pretrained_model.classifier[2],
            pretrained_model.classifier[3],
            pretrained_model.classifier[4],
            pretrained_model.classifier[5],
            additional_layer1,
            additional_layer2,
            additional_layer3,
            additional_layer4,
            additional_layer5,
            additional_layer6,
            additional_layer7,
            additional_layer8,
            additional_layer9,
            additional_layer10
        )
        # Get trainable layer names without weights or biases
        trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in pretrained_model.named_parameters() if param.requires_grad])))
        if modelweightpath is not None:
            pretrained_model.load_state_dict(torch.load(modelweightpath))
        return pretrained_model,trainable_layer_names
