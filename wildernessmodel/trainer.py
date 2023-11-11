import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import copy
class ANTHROPROTECTBinaryModel:
    def __init__(self, model, train_dataset, val_dataset, test_dataset,output_channels=2, trainbatch_size=8,valbatch_size=8,testbatch_size=16, num_epochs=2, learning_rate=0.001, log_dir='logs',lr_step_size=10, lr_gamma=0.5):
        """
        Args:
            model (_type_): _description_
            train_dataset (_type_): _description_
            val_dataset (_type_): _description_
            test_dataset (_type_): _description_
            output_channels (int, optional): _description_. Defaults to 1.
            batch_size (int, optional): _description_. Defaults to 8.
            num_epochs (int, optional): _description_. Defaults to 2.
            learning_rate (float, optional): _description_. Defaults to 0.001.
            log_dir (str, optional): _description_. Defaults to 'logs'.
            lr_step_size (int, optional): _description_. Defaults to 10.
            lr_gamma (float, optional): _description_. Defaults to 0.5.
        """
        self.model = copy.deepcopy(model)
        self.train_dataset = copy.deepcopy(train_dataset)
        self.val_dataset = copy.deepcopy(val_dataset)
        self.test_dataset = copy.deepcopy(test_dataset)
        self.trainbatch_size = trainbatch_size
        self.valbatch_size=valbatch_size
        self.testbatch_size=testbatch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_channels=output_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define loss function and optimizer
        if output_channels==1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.functional.cross_entropy
        
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_gamma)
        
        # Move the model to the appropriate device (CPU or GPU)
        self.model.to(self.device)
        
        # Create a directory for storing logs and checkpoints
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard SummaryWriter for logging
        self.writer = SummaryWriter(logdir=self.log_dir)
        self.writer.add_text('config/model', self.model.__class__.__name__)
        self.writer.add_text('config/model_details', str(self.model))
        self.writer.add_text('config/optimizer', self.optimizer.__class__.__name__)
        self.writer.add_text('config/scheduler', self.lr_scheduler.__class__.__name__)
        self.writer.add_text('config/cuda', str(self.device))
        self.writer.add_text('model', str(self.model))


    def calculate_class_weights(self, dataset):
        """_summary_

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Calculate class weights based on class frequencies in the training set
        class_counts = torch.bincount(torch.tensor(dataset.labels))
        total_samples = float(len(dataset))
        class_weights = total_samples / (class_counts * len(class_counts))
        return class_weights
    
    def bncalculate_accuracy(self,output, target):
        return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()
    
    def calculate_accuracy(self,output, target):
        return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()
    
    def train(self):
        """_summary_
        """
        # Create data loaders for training and validation
        train_loader = DataLoader(self.train_dataset, batch_size=self.trainbatch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.valbatch_size, shuffle=False)
        self.writer.add_text('config/trainset', train_loader.dataset.__class__.__name__)
        self.writer.add_text('config/valset', val_loader.dataset.__class__.__name__)
        best_val_loss = float('inf')  # Initialize with a large value
        checkpoint_path = os.path.join(self.log_dir, 'best_model.pth')
        train_iter=0
        val_iter=0
        running_acc = 0
        running_acctotal = 0
        valrunning_loss=0
        correct = 0
        total = 0
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.model.train()  # Set model to training mode
            # Use tqdm to create a progress bar for training
            with tqdm(train_loader, unit="batch") as t:
                for i, data in enumerate(t):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Calculate loss
                    if self.output_channels==1:
                        outputs = self.model(inputs)
                    # Calculate loss
                        predicted = (outputs > 0.5).float()
                        loss = self.criterion(outputs, labels.unsqueeze(1).float())
                        running_acc += labels.size(0)
                        running_acctotal += (predicted == labels.unsqueeze(1).float()).sum().item()
                        
                    else:
                        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
                        # Get the index of the class with the highest probability
                        predicted = torch.argmax(probabilities, dim=1)
                        loss = self.criterion(outputs.float(), labels.long())
                        running_acc += (predicted == labels).sum().item()
                        running_acctotal += labels.size(0)
                    accuracy = 100 * running_acc / running_acctotal
                    # Backpropagation
                    loss.backward()
                    # Update weights
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Update the tqdm progress bar
                    train_iter=+1
                    self.writer.add_scalar(tag='Loss/train',scalar_value= running_loss/ (train_iter+1),global_step= train_iter)
                    self.writer.add_scalar(tag='Accuracy/train',scalar_value= accuracy,global_step= train_iter)
                    t.set_postfix(loss=f"{running_loss / (train_iter+1):.4f}",accuracy=f"{accuracy:.4f}")
                self.lr_scheduler.step()

            # Validation
            self.model.eval()  # Set model to evaluation mode
            
            # Use tqdm to create a progress bar for validation
            with torch.no_grad(), tqdm(val_loader, unit="batch") as t:
                for index,data in enumerate(t):
                    #if batch_counter >= 50:
                    #    break  # Exit the loop after 50 batches
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    if self.output_channels==1:
                        # Round predictions to 0 or 1
                        valloss = self.criterion(outputs, labels.unsqueeze(1).float())
                        predicted = (outputs > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels.unsqueeze(1).float()).sum().item()
                        #accuray=self.calculate_accuracy(outputs, labels.unsqueeze(1).float())
                        val_accuracy = 100 * correct / total
                    else:
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        # Get the index of the class with the highest probability
                        predicted = torch.argmax(probabilities, dim=1)
                        valloss = self.criterion(outputs.float(), labels.long())
                        # Count the number of correctly classified samples
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                        val_accuracy = 100 * correct / total
                    # Update the tqdm progress bar
                    valrunning_loss += valloss.item()
                    val_iter += 1
                    self.writer.add_scalar(tag='Loss/val',scalar_value= valrunning_loss/(val_iter+1), global_step=val_iter)
                    self.writer.add_scalar(tag='Accuracy/val',scalar_value= val_accuracy,global_step= val_iter)
                    t.set_postfix(val_accuracy=f"{val_accuracy:.2f}%")
                   
            # Log validation accuracy using TensorBoard SummaryWriter
            
            # Check if the current model has the lowest validation loss
            if running_loss < best_val_loss:
                best_val_loss = running_loss
                # Save the best model checkpoint
                torch.save(self.model.state_dict(), checkpoint_path)
    
    def evaluate(self):
        # Create data loader for evaluation
        test_loader = DataLoader(self.test_dataset, batch_size=self.testbatch_size, shuffle=False)
        self.writer.add_text('config/testset', test_loader.dataset.__class__.__name__)
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        test_iter=0
        # Use tqdm to create a progress bar for evaluation
        with torch.no_grad(), tqdm(test_loader, unit="batch") as t:
            for data in t:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                if self.output_channels==1:
                        # Round predictions to 0 or 1
                        #loss = self.criterion(outputs, labels.unsqueeze(1).float())
                        predicted = (outputs > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels.unsqueeze(1).float()).sum().item()
                        #accuray=self.calculate_accuracy(outputs, labels.unsqueeze(1).float())
                        test_accuracy = 100 * correct / total
                else:
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        # Get the index of the class with the highest probability
                        predicted = torch.argmax(probabilities, dim=1)
                        #loss = self.criterion(outputs.float(), labels.long())
                        # Count the number of correctly classified samples
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                        test_accuracy = 100 * correct / total
                # Round predictions to 0 or 1
                test_iter+=1
                #self.writer.add_scalar('Loss/test', valrunning_loss/(val_iter+1), val_iter)
                self.writer.add_scalar(tag='Accuracy/test',scalar_value= test_accuracy,global_step=test_iter)
                t.set_postfix(test_accuracy=f"{test_accuracy:.2f}%")
        print(f"Accuracy on evaluation data: {test_accuracy:.2f}%")
        self.writer.flush()
        self.writer.close()
