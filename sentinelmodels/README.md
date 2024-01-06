# Wilderness Image Classifier

This project focuses on classifying images related to wilderness into specific categories using machine learning techniques.

## Overview

To use these classes for data preprocessing:

- Import the classes `SentinelDataset` and `LCSDataset` from the provided Python file.
- Create instances of these classes by providing the necessary parameters as outlined in the docstrings of each class.
- Utilize the methods provided within these classes for data loading, normalization, and dataset preparation.

To build and load deep neural network model :

- Multi-label and Multi-class classification using modified PyTorch models (AlexNet, VGG16, ResNet18).
- Customizable model architectures with varying output channels.
- Factory method (`buildmodel`) for easy model instantiation based on user-defined parameters.

The  `WildernessClassifier` is a Python class designed to handle the training, validation, evaluation, and hyperparameter tuning of image classification models for AntroProtect data.




## Requirements

To run this classifier, you will need:

- Python (3.6 or higher)
- PyTorch
- NumPy
- scikit-learn
- tqdm
- wandb (Weights and Biases)

## Usage

1. **Setup:**

    - Install the necessary dependencies mentioned above.
    - Ensure that your dataset is structured appropriately and that the CSV file containing dataset information follows the required format.

2. **Initializing the Classifier:**

    ```python
    # Initialize the WildernessClassifier object
    classifier = WildernessClassifier(csv_filepath, root_dir)
    ```

3. **Training:**

    Config file format :
    ```json
    "root_dir": ,#path where the dataset is present 
    "datasplitfilename": "",#name of the csv file which contains the list of images which are divided into different datasets
    "n_classes": ,# number of classes to be classified
    "device": "cuda",# "cpu" or "cuda" where the model needs to train
    "inputimage_type": "",#type of the image "rgb"
    "modeltype": "",# type of model ["alexnet","vgg16","resnet18"]
    "modelweightspath": "",# location of the saved weight of the model
    "trainable": ,#if true all layer are trained else only the classifier layers is trained
    "epochs": ,#Number of epoch for training 
    "batchsize": ,# Batch size of each step
    "lr": ,#learning rate required by the optimizer
    "optimizer": "",# type of optimizer ["sgd","adam"]
    "gamma": ,#the value determined the penalty in focalloss, if 0, it becomes cross-entropy 
    "lrscheduler": "step_lr",# type learning rate scheduler ["step_lr","exponent_lr"]    
    "losstype": "focalloss",
    "lossweights": ,#class loss weight used for unbalanced datasets, if None, the value are calculated using sklearn,default [1,1]
    "project_name": "wilderness-or-not",
    "log_images": ,#if true images and their prediction are stored during inference (testing and validation steps) 
    "log_image_index": ,#index where the image should be stored
    "earlystop_patience": 3,# epochs for early stop 
    "earlystop_min_delta": 0.05, 
    ```

    ```python
    # Start training the model
    config = {...}  # Define your configuration parameters
    classifier.training(config)
    ```

4. **Evaluation:**

    ```python
    # Evaluate the trained model on the test dataset
    test_accuracy = classifier.evaluate()
    ```

5. **Hyperparameter Tuning:**

    Tuning config file:
    ```json
    'lr': {'distribution': 'uniform','min': 0,'max': 0.1,},

    'batchsize': {'distribution': 'q_log_uniform_values','q': 4,'min': 4,'max': 16,},

    "losstype":{"values":["focal loss","cross entropy"},
     
    "gamma":{'distribution': 'uniform','min': 0,'max': 2,},
     
    'classweights':{'values':[[1,1],[0.75,1.35]]} #using sklearn
     
    'optimizer': {'values': ['adam', 'sgd']},
    
    'lrscheduler':{'values':["step_lr","exponential_lr"]},

    'epoch': {'distribution': 'uniform','min': 1,'max': 4,}
    ```
    ```python
    # Perform hyperparameter tuning
    tuning_config = {...}  # Define hyperparameters for tuning
    classifier.hyperparameter_tuning(tuning_config)
    ```

## Additional Notes

- The `WildernessClassifier` provides methods for model training, evaluation, hyperparameter tuning, and more.
- Adjust configurations and hyperparameters based on your specific dataset and requirements.

## Credits

This codebase uses various libraries and techniques. Below are some notable mentions:

- PyTorch: [Link to PyTorch](https://pytorch.org/)
- Weights and Biases (wandb): [Link to wandb](https://wandb.ai/site)

## Model weigths

[Link to model weights](https://drive.google.com/drive/folders/1TFlZGTcmVg34UrBbvkduZnhN1wVU0pht?usp=drive_link)