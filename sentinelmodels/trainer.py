# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Instead of building the models from scratch, we use pretrained model with the capabitily to use different pretrained weight, the classifier module is change for out case.
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: The function are built based on the modules found in pytorch and wandb tutorials.


import os
import math
import wandb
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,ExponentialLR
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix,accuracy_score, precision_score, recall_score, f1_score


from sentinelmodels.pretrained_models import buildmodel
from sentinelmodels.preprocessing import SentinelDataset,LCSDataset
from sentinelmodels.customlosses import FocalLoss,MultiLabelFocalLoss


class WildernessClassifier:
    """
    A class for training, validating and evaluating the classification of wilderness images.

    Args:
        csv_filepath (str): The file path of the CSV file containing the dataset information.
        root_dir (str): The root directory of the dataset.
        n_classes (int, optional): The number of classes for classification. Defaults to 2.
        input_imagetype (str, optional): The type of input image, either "rgb" or "other". Defaults to "rgb".
        filter_label (list, optional): A list of labels to filter the dataset. Defaults to None.
        transform (callable, optional): Optional data transformation to be applied to the input image. Defaults to None.
        device (str, optional): The device to be used for computation, either "cpu" or "gpu". Defaults to None.
    """
    def __init__(self, csv_filepath, root_dir,n_classes=2,input_imagetype="rgb",filter_label=None,transform=None,device=None):
        self.csv_filepath=csv_filepath
        self.root_dir=root_dir
        self.input_imagetype=input_imagetype
        self.filter_label=filter_label #if the number is selected only those labels will be present in the dataset (used during evalution)
        self.transform=transform
        self.device=device
        self.n_classes=n_classes
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = SentinelDataset(csv_filepath, root_dir,input_imagetype,self.n_classes,self.filter_label,datasettype="train",device=self.device)
        self.val_dataset = SentinelDataset(csv_filepath, root_dir,input_imagetype,self.n_classes,self.filter_label,datasettype="val",device=self.device)
        self.test_dataset = SentinelDataset(csv_filepath, root_dir,input_imagetype,self.n_classes,self.filter_label,datasettype="test",device=self.device)
        self.model=None

    def build_dataset(self,dataset,batchsize):
        """
        Create a data loader for a given dataset with a specified batch size and shuffling.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to be loaded into the data loader.
            batchsize (int): The number of samples per batch.

        Returns:
            loader (torch.utils.data.DataLoader): The data loader containing the input dataset with the specified batch size and shuffling.
        """
        loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,shuffle=True)
        return loader
    
    def build_subset(self, dataset, batchsize, subsetsize=0.1, seed=69):
        """
        Create a subset of the dataset by randomly selecting a specified percentage of samples without replacement.
    
        Args:
            dataset (torch.utils.data.Dataset): The dataset from which to create a subset.
            batchsize (int): The number of samples per batch in the data loader.
            subsetsize (float, optional): The percentage of samples to include in the subset. Defaults to 0.1.
            seed (int, optional): The seed for random number generation. Defaults to 69.
    
        Returns:
            torch.utils.data.DataLoader: The data loader containing the subset of the dataset with the specified batch size.
        """
        np.random.seed(seed) 
        subsample_size = int(subsetsize * len(self.train_dataset))
        indices = np.random.choice(len(dataset), subsample_size, replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subsample_dataset, batch_size=batchsize)
        return loader
    

    def build_optimizer(self, optimizertype, learning_rate):
        """
        Builds an optimizer object based on the specified optimizer type and learning rate.

        Args:
            optimizertype (str): The type of optimizer to be used, either "sgd" or "adam".
            learning_rate (float): The learning rate to be used by the optimizer.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer object created based on the specified optimizer type and learning rate.
        """
        if optimizertype == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=learning_rate, momentum=0.9)
        elif optimizertype == "adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=learning_rate,weight_decay=0.0005)
        return optimizer
    

    def build_lrscheduler(self, optimizer, schedulertype):
        """
        Builds and returns a learning rate scheduler object based on the specified optimizer type and learning rate.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer object for which the learning rate scheduler is built.
            schedulertype (str): The type of learning rate scheduler to be used, either "step_lr" or "exponential_lr".

        Returns:
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler object based on the specified optimizer type and learning rate.
        """
        if schedulertype == "step_lr":
            scheduler = StepLR(optimizer, step_size=1, gamma=0.5)  # You can pass additional arguments as needed
        elif schedulertype == "exponential_lr":
            scheduler = ExponentialLR(optimizer, gamma=0.9)  # Additional arguments can be passed here as well
        return scheduler
    

    def get_loss(self, losstype, weight=None, gamma=None, size_average=False):
        """
        Returns a loss function based on the specified loss type.

        Args:
            losstype (str): The type of loss function to be used. It can be "crossentropy", "focalloss", or any other value.
            weight (Tensor, optional): The weight tensor to be used for the loss function. Defaults to None.
            gamma (float, optional): The gamma value to be used for the focal loss function. Defaults to None.
            size_average (bool, optional): Whether to average the loss over the batch. Defaults to False.

        Returns:
            loss_fn (torch.nn.Module): The loss function object based on the specified loss type.
        """
        if losstype == "crossentropy":
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
            return loss_fn
        elif losstype == "focalloss":
            loss_fn = FocalLoss(alpha=weight, gamma=gamma)
            return loss_fn
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight)
            return loss_fn
        
    def calculate_metrics(self, prediction, label, mode="multiclass", tol=0.5):
        """
        Calculate various evaluation metrics such as accuracy, precision, recall, and F1-score for a given set of predictions and labels.

        Args:
            prediction (torch.Tensor): The predicted values for the samples.
            label (torch.Tensor): The true labels for the samples.
            mode (str, optional): The mode of classification, either "multiclass" or "binary". Defaults to "multiclass".
            tol (float, optional): The tolerance threshold for binary classification. Defaults to 0.5.

        Returns:
            dict: A dictionary containing the calculated evaluation metrics. The keys are "Accuracy", "Precision", "Recall", and "F1-score".

        Raises:
            None

        Example:
            # Initialize the WildernessClassifier object
            classifier = WildernessClassifier(csv_filepath, root_dir)

            # Calculate metrics for multiclass classification
            metrics_multiclass = classifier.calculate_metrics(predictions, labels, mode="multiclass")

            # Calculate metrics for binary classification
            metrics_binary = classifier.calculate_metrics(predictions, labels, mode="binary")
        """

        if mode == "multiclass":
            if len(prediction.shape) > 1:  # Convert one-hot encoded predictions to class labels
                prediction = torch.argmax(torch.softmax(prediction, dim=-1), dim=-1)
            if len(label.shape) > 1:  # Convert one-hot encoded targets to class labels
                label = torch.argmax(label, dim=1)

            y_true = prediction.cpu().numpy()
            y_pred = label.cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            metrics_dict = {'Accuracy': accuracy * 100, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
            return metrics_dict
        else:
            prediction = (torch.sigmoid(prediction) >= tol).float()
            TP = ((prediction == 1) & (label == 1)).sum().item()
            FP = ((prediction == 1) & (label == 0)).sum().item()
            TN = ((prediction == 0) & (label == 0)).sum().item()
            FN = ((prediction == 0) & (label == 1)).sum().item()

            # Calculate Accuracy
            accuracy = (TP + TN) / (TP + FP + TN + FN)

            # Calculate Precision
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0

            # Calculate Recall (Sensitivity)
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            # Calculate F1 Score
            f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics_dict = {'Accuracy': accuracy * 100, 'Precision': precision, 'Recall': recall, 'F1-score': f1score}
            return metrics_dict
    
    def load_model(self, modeltype, n_classes=2, trainable=False, modelweights=None, pretrained_weights="IMAGENET1K_V1"):
        """
        Load a pre-trained model based on the specified model type, number of classes, and other parameters.

        Args:
            modeltype (str): The type of model architecture to load.
            n_classes (int): The number of classes for the classification task. Default is 2.
            trainable (bool): Whether to set the feature layers as trainable or frozen. Default is False.
            modelweights (str): The path to custom model weights. Default is None.
            pretrained_weights (str): The weights to use for the pre-trained model. Default is "IMAGENET1K_V1".

        Returns:
            None
        """
        self.model = buildmodel(modeltype, n_classes, parameterstrainable=trainable, modelweightpath=modelweights, pretrained_weights=pretrained_weights).to(self.device)
        pass
    def hyperparameter_tuning(self, config=None, project_name="wilderness-classification", pretrained=False):
        """
        Perform hyperparameter tuning for the wilderness image classification model.

        Args:
            config (dict): A dictionary containing the hyperparameters for the hyperparameter tuning process.
            project_name (str): The name of the project for logging and tracking in wandb.
            pretrained (bool): Whether to use pretrained weights for the model.

        Returns:
            None
        """
        with wandb.init(config=config, project=project_name):
            config = wandb.config
            if config.modelweightspath == '':
                pretrainedweights = None
            if isinstance(config.lossweights, list):
                classweights = torch.Tensor(config.lossweights).to(self.device)
            self.load_model(config.modeltype, n_classes=self.n_classes, trainable=pretrained, modelweights=pretrainedweights)
            htdataloader = self.build_subset(self.train_dataset, config.batchsize, config.subsetsize, config.seed)
            optimizer = self.build_optimizer(config.optimizer, config.lr)
            learningrate_scheduler = self.build_lrscheduler(optimizer, config.lrscheduler)
            loss_fn = self.get_loss(config.losstype, classweights, config.gamma)
            running_loss = 0
            running_accuracy = 0
            step_iterator = 0
            total_samples = 0
            for epoch in range(config.epochs):
                with tqdm(htdataloader, unit="batch") as t:
                    for _, (images, labels) in enumerate(t):
                        step_iterator += 1
                        optimizer.zero_grad()
                        output = self.model(images)
                        loss = loss_fn(output, labels)
                        loss.backward()
                        optimizer.step()
                        metrics = self.calculate_metrics(output, labels)
                        running_loss += loss.item()
                        running_accuracy += metrics["Accuracy"]
                        total_samples += labels.size(0)
                        metrics["batch loss"] = loss.item()
                        wandb.log({"batch loss": loss.item()})
                        t.set_postfix(metrics)
                wandb.log({"loss": running_loss / step_iterator, "accuracy": running_accuracy / step_iterator, "epoch": epoch})
                learningrate_scheduler.step()
        pass

    
    def training(self, config, project_name="wilderness-classification", train=True, log_images=True, batch_idx=10, patience=2, min_delta=0.05, pretrained_weights="IMAGENET1K_V1"):
        """
        Trains a wilderness image classifier using the specified configuration parameters.

        Args:
            config (dict): A dictionary containing the configuration parameters for training the model.
            project_name (str, optional): The name of the Weights and Biases project. Defaults to "wilderness-classification".
            train (bool, optional): Whether to train the model. Defaults to True.
            log_images (bool, optional): Whether to log images using Weights and Biases. Defaults to True.
            batch_idx (int, optional): The index of the batch to log images. Defaults to 10.
            patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 2.
            min_delta (float, optional): The minimum change in validation loss required to be considered as improvement. Defaults to 0.05.
            pretrained_weights (str, optional): The path to the pretrained weights. Defaults to "IMAGENET1K_V1".

        Returns:
            None
        """
        with wandb.init(config=config, project=project_name):
            config = wandb.config
            if config.modelweightspath == '':
                self.load_model(config.modeltype, n_classes=self.n_classes, trainable=config.trainable, modelweights=None, pretrained_weights=pretrained_weights)
            else:
                self.load_model(config.modeltype, n_classes=self.n_classes, trainable=config.trainable, modelweights=wandb.config.modelweightspath, pretrained_weights=pretrained_weights)
            self.classweights = torch.Tensor(self.train_dataset.class_weight).to(self.device)
            if config.lossweights is not None:
                if isinstance(config.lossweights, list):
                    self.classweights = torch.Tensor(config.lossweights).to(self.device)
            print(self.classweights)
            early_stopping_counter = 0
            best_val_loss = float('inf')
            train_dl = self.build_dataset(self.train_dataset, config.batchsize)
            print(self.model.trainable_layer_names)
            optimizer = self.build_optimizer(config.optimizer, config.lr)
            learningrate_scheduler = self.build_lrscheduler(optimizer, config.lrscheduler)
            n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batchsize)
            loss_fn = self.get_loss(config.losstype, self.classweights, config.gamma)
            self.running_valloss = 0.
            self.running_valaccuracy = 0
            self.step_valiterator = 0
            self.total_valsamples = 0
            running_loss = 0
            running_accuracy = 0
            step_iterator = 0
            total_samples = 0
            for epoch in range(config.epochs):
                self.model.train()
                with tqdm(train_dl, unit="batch") as t:
                    for step, (images, labels) in enumerate(t):
                        step_iterator += 1
                        optimizer.zero_grad()
                        output = self.model(images)
                        if train:
                            loss = loss_fn(output, labels)
                            loss.backward()
                            optimizer.step()
                        metrics = self.calculate_metrics(output, labels)
                        running_loss += loss.item()
                        running_accuracy += metrics["Accuracy"]
                        total_samples += labels.size(0)
                        metrics["Loss"] = loss.item()
                        metrics["running_loss"] = running_loss / step_iterator
                        metrics["running_accuracy"] = running_accuracy / step_iterator
                        metrics["epoch"] = (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch
                        train_metric = {f'train_{key}': value for key, value in metrics.items()}
                        wandb.log(train_metric)
                        t.set_postfix(metrics)
                #learningrate_scheduler.step()
                #my_lr = learningrate_scheduler.get_lr()
# or
                #my_lr = learningrate_scheduler.optimizer.param_groups[0]['lr']
                my_lr=config.lr
                epoch_loss = running_loss / step_iterator
                epoch_accuracy = running_accuracy / step_iterator
                val_loss, val_accuracy = self.valid_model(log_images, batch_idx)
                t.write(f"Epoch [{epoch+1}/{config.epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy :.2f}%, Val Accuracy: {val_accuracy :.2f}%, learning Rate:{my_lr:.4f}")
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    folder_path = os.path.join(wandb.run.dir, config.modeltype)
                    os.makedirs(folder_path, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(folder_path, "best_model.pth"))
                    wandb.save("best_model.pth")
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print(f"Stopping early at epoch {epoch+1}")
                    break
                if epoch_loss < best_val_loss:
                    best_val_loss = running_loss
                    folder_path = os.path.join(wandb.run.dir, config.modeltype)
                    os.makedirs(folder_path, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(folder_path, "best_model.pth"))
                    wandb.save("best_model.pth")

            wandb.summary['test_accuracy'] = self.evaluate(log_images, batch_idx)
            wandb.finish()

    def valid_model(self, log_images=True, batch_idx=10):
        """
        Evaluate the trained model on the validation dataset.

        Args:
            log_images (bool, optional): A flag indicating whether to log the images for a specified batch index. Defaults to True.
            batch_idx (int, optional): The index of the batch for which to log the images. Defaults to 10.

        Returns:
            tuple: A tuple containing the average validation loss and average validation accuracy.

        """
        self.model.eval()
        valid_dl = self.build_dataset(self.val_dataset, wandb.config.batchsize * 2)
        loss_fn = self.get_loss(wandb.config.losstype, self.classweights, wandb.config.gamma)

        with torch.inference_mode():
            with tqdm(valid_dl, unit="batch") as t:
                for i, (images, labels) in enumerate(t):
                    self.step_valiterator += 1
                    valoutput = self.model(images)
                    valoss = loss_fn(valoutput, labels)
                    metrics = self.calculate_metrics(valoutput, labels)
                    self.running_valloss += valoss.item()
                    self.running_valaccuracy += metrics["Accuracy"]
                    metrics["Loss"] = valoss.item()
                    metrics["running_loss"] = self.running_valloss / self.step_valiterator
                    metrics["running_accuracy"] = self.running_valaccuracy / self.step_valiterator
                    metrics["globalstep"] = self.step_valiterator
                    valid_metric = {f'valid_{key}': value for key, value in metrics.items()}
                    if i == batch_idx and log_images:
                        print("logging images")
                        self.log_image_table(images, valoutput, labels)
                    wandb.log(valid_metric)
                    t.set_postfix(metrics)
        return self.running_valloss / self.step_valiterator, self.running_valaccuracy / self.step_valiterator


    def evaluate(self, log_images=False, batch_idx=50, config=None, project_name=None):
        """
        Evaluate the performance of the trained model on the test dataset.

        Args:
            log_images (bool, optional): Flag indicating whether to log images during evaluation. Defaults to False.
            batch_idx (int, optional): Index of the batch at which to log images. Defaults to 50.
            config (dict, optional): Dictionary containing the configuration parameters for the evaluation. Defaults to None.
            project_name (str, optional): Name of the project for logging. Defaults to None.

        Returns:
            float: Average test accuracy.
        """
        if config is not None:
            wandb.init(project=project_name, config=config)
            if wandb.config.modelweightspath == '':
                self.load_model(wandb.config.modeltype, n_classes=self.n_classes, trainable=False, modelweights=None)
            else:
                self.load_model(wandb.config.modeltype, n_classes=self.n_classes, trainable=False, modelweights=wandb.config.modelweightspath)
    
        self.model.eval()
        test_dl = self.build_dataset(self.test_dataset, wandb.config.batchsize * 2)
        running_testaccuracy = 0
        step_testiterator = 0
        running_testaccuracy = 0
        predicted = []
        groundtruth = []
    
        with torch.inference_mode():
            for i, (images, labels) in enumerate(test_dl):
                step_testiterator += 1
                testoutput = self.model(images)
                metrics = self.calculate_metrics(testoutput, labels)
                running_testaccuracy += metrics["Accuracy"]
            
                if i == batch_idx and log_images:
                    self.log_image_table(images, testoutput, labels)
            
                metrics["running_accuracy"] = running_testaccuracy / step_testiterator
                metrics["globalstep"] = step_testiterator
                test_metric = {f'test_{key}': value for key, value in metrics.items()}
                probability = torch.softmax(testoutput, dim=-1)
                predicted.extend(torch.argmax(probability, dim=-1).to("cpu").tolist())
                groundtruth.extend(labels.to("cpu").tolist())
                wandb.log(test_metric)
        
            cm = confusion_matrix(y_true=groundtruth, y_pred=predicted)
            print(cm)
            wandb.log({"conf_mat": wandb.plot.confusion_matrix(
                        y_true=groundtruth, preds=predicted,
                        class_names=wandb.config.classnames)})
        
            return running_testaccuracy / step_testiterator
    
    def log_image_table(self, images, valoutput, labels):
        """
        Log a table of images, predictions, and labels to the Weights and Biases (wandb) platform.

        Args:
            images (torch.Tensor): The input images.
            valoutput (torch.Tensor): The predicted values for the images.
            labels (torch.Tensor): The true labels for the images.

        Returns:
            None. The method logs the image table to the Weights and Biases platform.
        """
        # Create a wandb Table to log images, labels, and predictions
        probability = torch.softmax(valoutput, dim=-1)
        predicted = torch.argmax(probability, dim=-1)
        table = wandb.Table(columns=["image", "pred", "target"])
    
        # Iterate over the images, predicted labels, and true labels
        for img, pred, targ in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu")):
            table.add_data(wandb.Image(img.permute(1, 2, 0).numpy()), pred, targ)
    
        # Log the table to the Weights and Biases platform
        wandb.log({"predictions_table": table}, commit=False)

    def create_confusion_matrix(self, valoutput, labels, class_names=["Anthropogenic", "wilderness"]):
        """
        Create and log a confusion matrix using the wandb library.

        Args:
            valoutput (torch.Tensor): The output of the model for the validation data.
            labels (torch.Tensor): The true labels for the validation data.
            class_names (list, optional): A list of class names to be used for labeling the confusion matrix. Defaults to ["Anthropogenic", "wilderness"].

        Returns:
            None

        Example Usage:
            classifier = WildernessClassifier(csv_filepath, root_dir)
            valoutput = classifier.model(val_data)
            classifier.create_confusion_matrix(valoutput, labels, class_names=["Anthropogenic", "wilderness"])
        """
        probability = torch.softmax(valoutput, dim=-1)
        predicted = torch.argmax(probability, dim=-1)
        wandb.log({"conf_mat": wandb.plot.confusion_matrix(
            y_true=labels.to("cpu").tolist(), preds=predicted.to("cpu").tolist(),
            class_names=class_names)})
        pass




class WildernessMultlabelClassifier(WildernessClassifier):
    """
            The `WildernessMultlabelClassifier` class is used for training, validating, and evaluating the classification of wilderness images. It provides methods for training the model, evaluating its performance on validation and test datasets, and logging the results to the Weights and Biases platform.

            Example Usage:
                classifier = WildernessMultlabelClassifier(csv_filepath, root_dir)
                classifier.train_model()
                classifier.evaluate()

            Methods:
                train_model(): Trains the model using the specified dataset and optimizer. Logs the training metrics to the Weights and Biases platform.
                valid_model(log_images=True, batch_idx=10): Evaluates the trained model on the validation dataset. Calculates the validation loss and accuracy. Logs the metrics to the Weights and Biases platform. Optionally logs a table of images, predictions, and labels.
                evaluate(log_images=False, batch_idx=50, config=None, project_name=None): Evaluates the performance of the trained model on the test dataset. Calculates the test accuracy. Logs the metrics to the Weights and Biases platform. Optionally logs a confusion matrix and a table of images, predictions, and labels.
                log_image_table(images, valoutput, labels): Logs a table of images, predictions, and labels to the Weights and Biases platform.
                create_confusion_matrix(valoutput, labels, class_names=["Anthropogenic", "wilderness"]): Creates and logs a confusion matrix using the Weights and Biases platform.

            Fields:
                running_valaccuracy: Running validation accuracy.
                step_valiterator: Step iterator for validation.
                total_valsamples: Total number of validation samples.
                running_loss: Running loss during training.
                running_accuracy: Running accuracy during training.
                step_iterator: Step iterator during training.
                total_samples: Total number of samples during training.
            """
    def __init__(self, csv_filepath, root_dir,n_classes=2,nlabels=5,input_imagetype="rgb",modeltype="multilabel",channel_name="corine",filter_label=None,transform=None,device=None):
        self.csv_filepath=csv_filepath
        self.root_dir=root_dir
        self.input_imagetype=input_imagetype
        self.filter_label=filter_label
        self.transform=transform
        self.device=device
        self.nlabels=nlabels
        self.n_classes=n_classes
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset=LCSDataset(csv_filepath, root_dir,"train",input_imagetype,modeltype,channel_name,filter_label=self.filter_label,transform=self.transform,device=self.device)
        self.val_dataset = LCSDataset(csv_filepath, root_dir,"val",input_imagetype,modeltype,channel_name,filter_label=self.filter_label,transform=self.transform,device=self.device)
        self.test_dataset = LCSDataset(csv_filepath, root_dir,"test",input_imagetype,modeltype,channel_name,filter_label=self.filter_label,transform=self.transform,device=self.device)
        self.model=None
    

    def load_model(self,modeltype,n_classes=2,multilabel=5,trainable=False,modelweights=None,pretrained_weights="IMAGENET1K_V1"):
        self.model=buildmodel(modeltype,multilabel,None,parameterstrainable=trainable,modelweightpath=modelweights,pretrained_weights=pretrained_weights).to(self.device)
        pass
    

    def calculate_metrics(self,prediction,label,mode="multilabel",tol=0.5):
        return super().calculate_metrics(prediction,label,mode,tol)
    

    def get_loss(self,losstype,weight=None,gamma=2,size_average=False):
        if losstype=="multilabelloss":
            return MultiLabelFocalLoss(gamma=gamma,alpha=weight)
        return super().get_loss(losstype,weight,gamma,size_average)
    

    def get_labeldict(self, indices):
        """
        Creates a dictionary that maps each key to a list of values.

        Args:
            indices (list): A list of indices, where each index is a tuple of two elements.

        Returns:
            dict: A dictionary where each key is mapped to a list of values.
        """
        dct = {}
        for idx in indices:
            key = idx[0].item()
            value = idx[1].item()
            if key in dct:
                dct[key].append(value)
            else:
                dct[key] = [value]
        return dct
        

    def log_image_table(self, images, predicted, labels, tol=0.5):
        """
        Log a wandb.Table with image, predicted label, target label, and scores.

        Args:
            images (torch.Tensor): A tensor containing the input images.
            predicted (torch.Tensor): A tensor containing the predicted labels for the images.
            labels (torch.Tensor): A tensor containing the true labels for the images.
            tol (float, optional): A float value representing the tolerance threshold for binary classification. Defaults to 0.5.

        Returns:
            None
        """
        # 🐝 Create a wandb Table to log images, labels and predictions to
        probability = (torch.sigmoid(predicted) > tol).float()
        pred_indices = self.get_labeldict(torch.nonzero(probability == 1, as_tuple=False))
        target_indices = self.get_labeldict(torch.nonzero(labels == 1, as_tuple=False))

        table = wandb.Table(columns=["image", "pred 0", "target 0", "pred 1", "target 1", "pred 2", "target 2", "pred 3", "target 3", "targetlist", "predictionlist"])
        for img, pred, targ in zip(images.to("cpu"), probability.to("cpu"), labels.to("cpu")):
            _, pred_index = pred.topk(4)
            _, target_index = targ.topk(4)
            table.add_data(wandb.Image(img.permute(1, 2, 0).numpy()), pred_index[0], target_index[0], pred_index[1], target_index[1], pred_index[2], target_index[2], pred_index[3], target_index[3], target_indices, pred_indices)
        wandb.log({"predictions_table": table}, commit=False)


    def training(self,config,project_name="wilderness-classification",train=True,log_images=True,batch_idx=10,patience=2, min_delta=0.05,pretrained_weights="IMAGENET1K_V1"):
        with wandb.init(config=config,project=project_name):
            config = wandb.config
            if config.modelweightspath=='':
               self.load_model(config.modeltype,n_classes=self.n_classes,multilabel=self.nlabels,trainable=config.trainable,modelweights=None,pretrained_weights=pretrained_weights)
            
            else:
                self.load_model(config.modeltype,n_classes=self.n_classes,multilabel=self.nlabels,trainable=config.trainable,modelweights=wandb.config.modelweightspath,pretrained_weights=pretrained_weights)
            self.classweights=torch.Tensor(self.train_dataset.class_weight).to(self.device)
            if config.lossweights is not None:
                if isinstance(config.lossweights,list):
                    self.classweights=torch.Tensor(config.lossweights).to(self.device)
            print(self.classweights)
            early_stopping_counter = 0
            best_val_loss = float('inf')
            train_dl=self.build_dataset(self.train_dataset,config.batchsize)
            print(self.model.trainable_layer_names)
            optimizer=self.build_optimizer(config.optimizer,config.lr)
            learningrate_scheduler=self.build_lrscheduler(optimizer,config.lrscheduler)
            n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batchsize)
            loss_fn=self.get_loss(config.losstype,self.classweights,config.gamma)
            self.running_valloss = 0.
            self.running_valaccuracy=0
            self.step_valiterator=0
            self.total_valsamples=0
            running_loss=0
            running_accuracy=0
            step_iterator=0
            total_samples=0
            halfway_step=int(len(train_dl)/2)
            for epoch in range(config.epochs):
                self.model.train()
                with tqdm(train_dl, unit="batch") as t:
                    for step, (images,labels) in enumerate(t):
                        step_iterator+=1
                        optimizer.zero_grad() 
                        output= self.model(images)
                        if train:
                            loss=loss_fn(output,labels)
                            loss.backward()
                            optimizer.step()  
                            if step == halfway_step:
                                learningrate_scheduler.step()
                        metrics=self.calculate_metrics(output,labels)
                        running_loss+=loss.item()
                        running_accuracy+=metrics["Accuracy"]
                        total_samples += labels.size(0)
                        metrics["Loss"]=loss.item()
                        metrics["running_loss"]=running_loss/step_iterator
                        metrics["running_accuracy"]=running_accuracy/step_iterator
                        metrics["epoch"]=(step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch
                        train_metric = {f'train_{key}': value for key, value in metrics.items()}
                        wandb.log(train_metric)
                        t.set_postfix(metrics)
                learningrate_scheduler.step()
                epoch_loss = running_loss / step_iterator
                epoch_accuracy = running_accuracy / step_iterator
                val_loss, val_accuracy =self.valid_model(log_images,batch_idx)
                t.write(f"Epoch [{epoch+1}/{config.epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy :.2f}%, Val Accuracy: {val_accuracy :.2f}%")
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    folder_path=os.path.join(wandb.run.dir,config.modeltype)
                    os.makedirs(folder_path, exist_ok=True)
                    torch.save(self.model.state_dict(),os.path.join(folder_path, "best_model.pth"))
                    wandb.save("best_model.pth")
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print(f"Stopping early at epoch {epoch+1}")
                    break
                if epoch_loss < best_val_loss:
                    best_val_loss = running_loss
                    folder_path=os.path.join(wandb.run.dir,config.modeltype)
                    os.makedirs(folder_path, exist_ok=True)
                    torch.save(self.model.state_dict(),os.path.join(folder_path, "best_model.pth"))
                    wandb.save("best_model.pth")

            wandb.summary['test_accuracy'] = self.evaluate(log_images,batch_idx)   
            wandb.finish()

    def valid_model(self, log_images=True, batch_idx=10):
        """
        Evaluate the performance of the trained model on the validation dataset.

        Args:
            log_images (bool, optional): Flag indicating whether to log a table of images, predictions, and labels. Default is True.
            batch_idx (int, optional): The index of the batch at which to log the images table. Default is 10.

        Returns:
            tuple: A tuple containing the average validation loss and average validation accuracy.

        """
        self.model.eval()
        valid_dl = self.build_dataset(self.val_dataset, wandb.config.batchsize * 2)
        loss_fn = self.get_loss(wandb.config.losstype, self.classweights, wandb.config.gamma)

        with torch.inference_mode():
            with tqdm(valid_dl, unit="batch") as t:
                for i, (images, labels) in enumerate(t):
                    self.step_valiterator += 1
                    valoutput = self.model(images)
                    valoss = loss_fn(valoutput, labels)
                    metrics = self.calculate_metrics(valoutput, labels)
                    self.running_valloss += valoss.item()
                    self.running_valaccuracy += metrics["Accuracy"]
                    metrics["Loss"] = valoss.item()
                    metrics["running_loss"] = self.running_valloss / self.step_valiterator
                    metrics["running_accuracy"] = self.running_valaccuracy / self.step_valiterator
                    metrics["globalstep"] = self.step_valiterator
                    valid_metric = {f'valid_{key}': value for key, value in metrics.items()}
                    if i == batch_idx and log_images:
                        print("logging images")
                        self.log_image_table(images, valoutput, labels)
                    wandb.log(valid_metric)
                    t.set_postfix(metrics)
        return self.running_valloss / self.step_valiterator, self.running_valaccuracy / self.step_valiterator


    def evaluate(self, log_images=False, batch_idx=50, config=None, project_name=None):
        """
        Evaluate the performance of the trained model on the test dataset.

        Args:
            log_images (bool, optional): Whether to log a table of images, predictions, and labels. Default is False.
            batch_idx (int, optional): The index of the batch to log the images. Default is 50.
            config (object, optional): The configuration object containing the model settings. Default is None.
            project_name (str, optional): The name of the project in the Weights and Biases platform. Default is None.

        Returns:
            float: The test accuracy.

        Example Usage:
            classifier = WildernessMultlabelClassifier(csv_filepath, root_dir)
            classifier.load_model(modeltype, n_classes, multilabel, trainable, modelweights, pretrained_weights)
            classifier.evaluate(log_images, batch_idx, config, project_name)
        """
        if config is not None:
            wandb.init(project=project_name, config=config)
            if config.modelweightspath == '':
                self.load_model(config.modeltype, n_classes=self.n_classes, multilabel=self.nlabels, trainable=config.trainable, modelweights=None)
            else:
                self.load_model(config.modeltype, n_classes=self.n_classes, multilabel=self.nlabels, trainable=config.trainable, modelweights=wandb.config.modelweightspath)
    
        self.model.eval()
        test_dl = self.build_dataset(self.test_dataset, wandb.config.batchsize*2)
        running_testaccuracy = 0
        step_testiterator = 0
        running_testaccuracy = 0
        predicted = []
        groundtruth = []
    
        with torch.inference_mode():
            for i, (images, labels) in enumerate(test_dl):
                step_testiterator += 1
                testoutput = self.model(images)
                metrics = self.calculate_metrics(testoutput, labels)
                running_testaccuracy += metrics["Accuracy"]
            
                if i == batch_idx and log_images:
                    self.log_image_table(images, testoutput, labels)
            
                metrics["running_accuracy"] = running_testaccuracy / step_testiterator
                metrics["globalstep"] = step_testiterator
                test_metric = {f'test_{key}': value for key, value in metrics.items()}
                probability = torch.sigmoid(testoutput)
                predicted.append((probability>0.5).float().to("cpu").tolist())
                groundtruth.append(labels.to("cpu").tolist())
                wandb.log(test_metric)
            cm=multilabel_confusion_matrix(np.array(predicted), np.array(groundtruth))
            print(cm)
            wandb.log({"conf_mat": wandb.plot.confusion_matrix(
                        y_true=groundtruth, preds=predicted,
                        class_names=wandb.config.classnames)})
            return running_testaccuracy / step_testiterator