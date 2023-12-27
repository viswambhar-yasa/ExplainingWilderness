# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Instead of building the models from scratch, we use pretrained model with the capabitily to use different pretrained weight, the classifier module is change for out case.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: The function are built based on the modules found in pytorch tutorials.


import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, output_channels,parameterstrainable=False, pretrained_weights="IMAGENET1K_V1", modelweightpath=None):
        super(AlexNet, self).__init__()
        self.output_channels = output_channels

        # Load the pre-trained AlexNet model
        pretrained_model = models.alexnet(weights=pretrained_weights)
        self.features = pretrained_model.features
        self.avgpool = pretrained_model.avgpool

        for param in pretrained_model.features.parameters():
            param.requires_grad = parameterstrainable

        # Additional layers for custom output
        additional_layers = [
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.output_channels)
        ]

        # Modify the model by replacing the original fully connected layer with the additional layers
        self.classifier = nn.Sequential(
            *pretrained_model.classifier[:-1],  # Use all layers except the last one
            *additional_layers
        )

        self.trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in self.named_parameters() if param.requires_grad])))

        # Load custom model weights if provided
        if modelweightpath is not None:
            self.load_state_dict(torch.load(modelweightpath))


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetMultiClassnLabel(nn.Module):
    """
    A PyTorch module that extends the functionality of the AlexNet model for multi-label and multi-class classification tasks.
    Args:
        multilabel_channels (int): The number of output channels for the multi-label classification task.
        multiclass_channels (int): The number of output channels for the multi-class classification task.
        parameterstrainable (bool, optional): Whether to set the feature layers as trainable or frozen. Defaults to False.
        pretrained_weights (str, optional): The weights to use for the pre-trained AlexNet model. Defaults to "IMAGENET1K_V1".
        modelweightpath (str, optional): The path to custom model weights. Defaults to None.
    Example:
        # Create an instance of AlexNetMultiClassnLabel
        model = AlexNetMultiClassnLabel(multilabel_channels=5, multiclass_channels=3, parameterstrainable=True, pretrained_weights="IMAGENET1K_V1", modelweightpath="model_weights.pth")
        # Forward pass
        inputs = torch.randn(1, 3, 224, 224)
        multilabel_output, multiclass_output = model(inputs)
        # Print the outputs
        print(multilabel_output.shape)  # (1, 5)
        print(multiclass_output.shape)  # (1, 3)
    """
    def __init__(self, multilabel_channels, multiclass_channels, parameterstrainable=False, pretrained_weights="IMAGENET1K_V1", modelweightpath=None):
        super(AlexNetMultiClassnLabel, self).__init__()
        self.multilabel_channels = multilabel_channels
        self.multiclass_channels = multiclass_channels

        # Load the pre-trained AlexNet model
        pretrained_model = models.alexnet(weights=pretrained_weights)

        # Set feature layers to be trainable
        for param in pretrained_model.features.parameters():
            param.requires_grad = parameterstrainable

        self.features = pretrained_model.features
        self.avgpool = pretrained_model.avgpool

        # Common layers for both multi-label and multi-class
        self.common_layers = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Separate output layers for multi-label and multi-class
        self.multilabel_output = nn.Linear(256, self.multilabel_channels)
        self.multiclass_output = nn.Linear(256, self.multiclass_channels)

        self.trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in self.named_parameters() if param.requires_grad])))

        # Load custom model weights if provided
        if modelweightpath is not None:
            self.load_state_dict(torch.load(modelweightpath))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.common_layers(x)

        # Compute both outputs
        multilabel_output = self.multilabel_output(x)
        multiclass_output = self.multiclass_output(x)
        return multilabel_output, multiclass_output



class VGG16(nn.Module):
    """
    A custom implementation of the VGG16 model.
    Args:
        output_channels (int): The number of output channels for the model.
        parameterstrainable (bool, optional): Whether the parameters of the model's features are trainable. Defaults to False.
        pretrained (str, optional): The type of pre-trained weights to load. Defaults to "IMAGENET1K_V1".
        modelweightpath (str, optional): The path to custom model weights. Defaults to None.
    Attributes:
        nclass_output (nn.Linear): The output layer for multi-class predictions.
        trainable_layer_names (list): The names of trainable layers in the model.
    Example Usage:
        model = VGG16(output_channels=10)
        input_tensor = torch.randn(1, 3, 224, 224)
        class_output = model(input_tensor)
        print(multiclass_output.shape)  # Output: torch.Size([1, 10])
    """
    def __init__(self, output_channels, parameterstrainable=False,pretrained="IMAGENET1K_V1" ,modelweightpath=None):
        super(VGG16, self).__init__()
        self.output_channels = output_channels

        # Load the pre-trained VGG16 model
        pretrained_model = models.vgg16(weights=pretrained)
        self.features = pretrained_model.features
        self.avgpool = pretrained_model.avgpool

        for param in pretrained_model.features.parameters():
            param.requires_grad = parameterstrainable

        # Additional layers for custom output
        additional_layers = [
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.output_channels)
        ]

        # Modify the model by replacing the original fully connected layer with the additional layers
        self.classifier = nn.Sequential(*additional_layers)

        # Get the names of trainable layers
        self.trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in pretrained_model.named_parameters() if param.requires_grad])))

        # Load custom model weights if provided
        if modelweightpath is not None:
            self.load_state_dict(torch.load(modelweightpath))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG16MultiClassnLabel(nn.Module):
    """
    A custom implementation of the VGG16 model for multi-label and multi-class classification tasks.

    Args:
        multilabel_channels (int): The number of output channels for the multi-label output.
        multiclass_channels (int): The number of output channels for the multi-class output.
        parameterstrainable (bool, optional): Whether to set the feature layers to be trainable or not. Defaults to False.
        pretrained (str, optional): The type of pre-trained weights to load. Defaults to "IMAGENET1K_V1".
        modelweightpath (str, optional): The path to custom model weights. Defaults to None.

    Example Usage:
        model = VGG16MultiClassnLabel(multilabel_channels=5, multiclass_channels=3)
        input_tensor = torch.randn(1, 3, 224, 224)
        multilabel_output, multiclass_output = model(input_tensor)
        print(multilabel_output.shape)  # Output: torch.Size([1, 5])
        print(multiclass_output.shape)  # Output: torch.Size([1, 3])
    """

    def __init__(self, multilabel_channels, multiclass_channels, parameterstrainable=False, pretrained="IMAGENET1K_V1", modelweightpath=None):
        super(VGG16MultiClassnLabel, self).__init__()
        self.multilabel_channels = multilabel_channels
        self.multiclass_channels = multiclass_channels

        # Load the pre-trained VGG16 model
        pretrained_model = models.vgg16(weights=pretrained)

        # Set feature layers to be trainable or not based on parameterstrainable
        for param in pretrained_model.features.parameters():
            param.requires_grad = parameterstrainable

        # Copy the features and avgpool from the pretrained model
        self.features = pretrained_model.features
        self.avgpool = pretrained_model.avgpool

        # Common layers for both multi-label and multi-class
        self.common_layers = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Separate output layers for multi-label and multi-class
        self.multilabel_output = nn.Linear(256, self.multilabel_channels)
        self.multiclass_output = nn.Linear(256, self.multiclass_channels)

        # Load custom model weights if provided
        if modelweightpath is not None:
            self.load_state_dict(torch.load(modelweightpath))

        # Get the names of trainable layers after potentially loading custom weights
        self.trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in self.named_parameters() if param.requires_grad])))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.common_layers(x)

        # Compute both outputs
        multilabel_output = self.multilabel_output(x)
        multiclass_output = self.multiclass_output(x)
        return multilabel_output, multiclass_output


class ResNet18(nn.Module):
    """
    ResNet18 is a custom implementation of the ResNet-18 model for image classification tasks.
    Args:
        output_channels (int): The number of output channels for the model.
        parameterstrainable (bool, optional): Whether to set the feature layers as trainable or not. Defaults to False.
        pretrained (str, optional): The pre-trained model to load. Defaults to "IMAGENET1K_V1".
        modelweightpath (str, optional): The path to custom model weights. Defaults to None.
    Example Usage:
        model = ResNet18(output_channels=10)
        input_tensor = torch.randn(1, 3, 224, 224)
        class_output = model(input_tensor)
        print(class_output.shape)  # Output: torch.Size([1, 10])
    """
    def __init__(self, num_classes, parameterstrainable=False, pretrained_weights="IMAGENET1K_V1", modelweightpath=None):
        super(ResNet18, self).__init__()
        self.output_channels = num_classes

        # Load the pre-trained ResNet-18 model
        pretrained_model = models.resnet18(weights=pretrained_weights)

        # Set feature layers to be trainable or not based on parameterstrainable
        for param in pretrained_model.parameters():
            param.requires_grad = parameterstrainable

        # Replace the fully connected layer of ResNet-50 for the specified number of classes
        num_features = pretrained_model.fc.in_features
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.common_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.output_channels)
        )
        # Load custom model weights if provided
        if modelweightpath is not None:
            self.load_state_dict(torch.load(modelweightpath))

        self.trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in self.named_parameters() if param.requires_grad])))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.common_layers(x)
        return x


class ResNet18MultiClassnLabel(nn.Module):
    """
    ResNet18MultiClassnLabel is a modified version of the ResNet18 class that supports multi-label classification tasks in addition to multi-class classification tasks.
    Args:
        num_classes (int): The number of output classes for the model.
        multiclass_channels (int): The number of output channels for the multi-class classification.
        multilabel_channels (int): The number of output channels for the multi-label classification.
        pretrained_weights (str, optional): The pre-trained model to load. Defaults to "IMAGENET1K_V1".
        modelweightpath (str, optional): The path to custom model weights. Defaults to None.
    Example Usage:
        model = ResNet18MultiClassnLabel(num_classes=10, multiclass_channels=3, multilabel_channels=5)
        input_tensor = torch.randn(1, 3, 224, 224)
        multilabel_output, multiclass_output = model(input_tensor)
        print(multilabel_output.shape)  # Output: torch.Size([1, 5])
        print(multiclass_output.shape)  # Output: torch.Size([1, 3])
    """
    def __init__(self, multilabel_channels, multiclass_channels, parameterstrainable=False, pretrained="IMAGENET1K_V1", modelweightpath=None):
        super(ResNet18MultiClassnLabel, self).__init__()
        self.multilabel_channels = multilabel_channels
        self.multiclass_channels = multiclass_channels

        # Load the pre-trained ResNet-18 model
        pretrained_model = models.resnet18(weights=pretrained)

        # Set feature layers to be trainable or not based on parameterstrainable
        for param in pretrained_model.parameters():
            param.requires_grad = parameterstrainable

        # Replace the fully connected layer of ResNet-18
        num_features = pretrained_model.fc.in_features
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])

        # Common layers for both multi-label and multi-class
        self.common_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Separate output layers for multi-label and multi-class
        self.multilabel_output = nn.Linear(256, self.multilabel_channels)
        #self.multiclass_output = nn.Linear(256, self.multiclass_channels)

        # Load custom model weights if provided
        if modelweightpath is not None:
            self.load_state_dict(torch.load(modelweightpath))

        # Get the names of trainable layers after potentially loading custom weights
        self.trainable_layer_names = sorted(list(set([name.rsplit('.', 1)[0] for name, param in self.named_parameters() if param.requires_grad])))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.common_layers(x)

        # Compute both outputs
        multilabel_output = self.multilabel_output(x)
        #multiclass_output = self.multiclass_output(x)
        return multilabel_output#, multiclass_output



def buildmodel(model_type="alexnet", multiclass_channels=3, multilabel_channels=None, parameterstrainable=False, pretrained_weights="IMAGENET1K_V1", modelweightpath=None):
    """
    Create and return an instance of a specified model architecture based on the given parameters.

    Args:
        model_type (str): The type of model architecture to build ('alexnet', 'vgg16', or 'resnet18').
        multiclass_channels (int): The number of output channels for the multi-class classification task.
        multilabel_channels (int, optional): The number of output channels for the multi-label classification task. Default is None.
        parameterstrainable (bool, optional): Whether to set the feature layers as trainable or frozen. Default is False.
        pretrained_weights (str, optional): The weights to use for the pre-trained model. Default is "IMAGENET1K_V1".
        modelweightpath (str, optional): The path to custom model weights. Default is None.

    Returns:
        model: An instance of the specified model architecture.

    Raises:
        ValueError: If an unsupported model type is specified.
    """
    if model_type == 'alexnet':
        if multilabel_channels is not None :
            return AlexNetMultiClassnLabel(multilabel_channels, multiclass_channels, parameterstrainable, pretrained_weights, modelweightpath)
        else:
            return AlexNet(multiclass_channels, parameterstrainable, pretrained_weights, modelweightpath)

    elif model_type == 'vgg16':
        if multilabel_channels is not None :
            return VGG16MultiClassnLabel(multilabel_channels, multiclass_channels, parameterstrainable, pretrained_weights, modelweightpath)
        else:
            return VGG16(multiclass_channels, parameterstrainable, pretrained_weights, modelweightpath)

    elif model_type == 'resnet18':
        if multilabel_channels is not None:
            return ResNet18MultiClassnLabel(multilabel_channels, multiclass_channels, parameterstrainable, pretrained_weights, modelweightpath)
        else:
            return ResNet18(multiclass_channels, parameterstrainable, pretrained_weights, modelweightpath)
    else:
        raise ValueError("Unsupported model type specified")
