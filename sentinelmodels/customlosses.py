# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Custom loss function used for imbalanced datasets
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: The function are built based on the cross-entropy loss implementation in pytorch.


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Based on Focal Loss for Efficient Object Detection

    This class implements the Focal Loss formula, which is designed to address the problem of class imbalance in object detection tasks.
    The loss function is defined as FL(pt) = -alpha * (1 - pt)^delta * log(pt), where pt is the predicted probability of the true class, alpha is a custom scaling weight for different classes, and delta is a hyperparameter to adjust the focus.

    Args:
        alpha (Tensor, optional): Custom scaling weights for different classes.
        delta (float, optional): Hyperparameter to adjust focus.
        average_loss (bool, optional): Determines if loss is averaged or summed over a batch.
    """
    def __init__(self, alpha=None, gamma=2, average_loss=False):
        """
        Initializes the FocalLoss object with the provided parameters.

        Args:
            alpha (Tensor, optional): Custom scaling weights for different classes.
            delta (float, optional): Hyperparameter to adjust focus.
            average_loss (bool, optional): Determines if loss is averaged or summed over a batch.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.average_loss = average_loss

    def forward(self, predictions, labels):
        """
        Calculates the focal loss based on the provided predictions and labels.

        Args:
            predictions (Tensor): The predicted probabilities for each class.
            labels (Tensor): The true labels for each sample.

        Returns:
            Tensor: The modulated loss, either averaged or summed based on the `average_loss` parameter.
        """
        probabilities = F.softmax(predictions, dim=1)
        cross_entropy = F.cross_entropy(predictions, labels, reduction='none', weight=self.alpha)#CE(pt) = −αt log(pt).
        pt = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        modulated_loss = ((1 - pt) ** self.gamma) * cross_entropy#FL(pt) = −(1 − pt)γ log(pt)
        if self.average_loss:
            return modulated_loss.mean()
        else:
            return modulated_loss.sum()

        

class MultiLabelFocalLoss(nn.Module):
    """
        Based on Focal Loss for Dense Object Detection Modified the initial implementation to work for multi labels problem.
        Args:
            gamma (float): The focusing parameter of the focal loss.
            alpha (Tensor, optional): The balancing parameter of the focal loss. Default: None.
            size_average (bool, optional): A boolean indicating whether to average the loss or sum it. Default: True.

        Example Usage:
            loss_fn = MultiLabelFocalLoss(gamma=2.0, alpha=None, size_average=True)
            predictions = torch.tensor([[0.8, 0.2], [0.6, 0.7]])
            labels = torch.tensor([1, 0],[1,1])
            loss = loss_fn(predictions, labels)
            print(loss)  # Output: tensor(0.4616)
        """
    def __init__(self, gamma=2.0, alpha=None, average_loss=False):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.average_loss = average_loss

    def forward(self, input, target):
        if self.alpha is not None:
            self.alpha = self.alpha.to(input.device)
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(input) #

        # Binary Cross Entropy Loss
         # Calculating pt
        BCE = F.binary_cross_entropy(probabilities, target,weight=self.alpha, reduction='none')#CE(pt) = −αt log(pt).
       
        # Focal loss components
        modulated_loss = ((1 - probabilities) ** self.gamma) * BCE#FL(pt) = −(1 − pt)γ log(pt)
        
        if self.size_average:
            return modulated_loss.mean()
        else:
            return modulated_loss.sum()