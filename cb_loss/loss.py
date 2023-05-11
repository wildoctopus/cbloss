import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    This is a PyTorch implementation of the Focal Loss, which is designed to address class
    imbalance problems in classification tasks by down-weighting the contribution of easy
    negative samples. The loss value is increased for misclassified examples, particularly
    those with low predicted probabilities and high true probabilities.

    Formula:
    For a binary classification problem:
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)                if y = 1
             - (1 - alpha) * pt**gamma * log(1 - pt)            otherwise

    For a multi-class classification problem:
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)                if y = c
             - (1 - alpha) * pt**gamma * log(1 - pt)            otherwise
    where:
    pt = sigmoid(x) for binary classification, and softmax(x) for multi-class classification
    alpha = balancing parameter, default to 1, the balance between positive and negative samples
    gamma = focusing parameter, default to 2, the degree of focusing, 0 means no focusing

    Reference:
    - Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
      Focal loss for dense object detection. In Proceedings of the IEEE international
      conference on computer vision (pp. 2980-2988).
    - https://arxiv.org/pdf/1708.02002.pdf

    Args:
        alpha (float): balancing parameter, default to 1.
        gamma (float): focusing parameter, default to 2.
        reduction (str): reduction method for the loss, either 'mean' or 'sum', default to 'mean'.

    Inputs:
        outputs (tensor): model predictions of shape (batch_size, num_classes) for multi-class,
                          or (batch_size, ) for binary-class.
        targets (tensor): ground truth labels of shape (batch_size, ) for binary-class,
                          or (batch_size, num_classes) for multi-class.

    Returns:
        Focal loss value.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        if len(targets.shape) == 1 or targets.shape[1] == 1:
            # binary classification problem
            ce_loss = F.binary_cross_entropy_with_logits(outputs, targets.float(), reduction='none')
        else:
            # multi-class classification problem
            ce_loss = F.cross_entropy(outputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt)**self.gamma * ce_loss)
        if self.reduction == 'none':
            return focal_loss 
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}. Please use 'mean' or 'sum'.")
        


class ClassBalancedLoss(nn.Module):
    """
    Class-balanced loss function for multi-class classification problems.

    This loss function helps address the problem of class imbalance in the training data by assigning
    higher weights to underrepresented classes during training. The weights are determined based on
    the number of samples per class and a beta value, which controls the degree of balancing between
    the classes.

    The loss function supports different types of base losses, including CrossEntropyLoss, BCEWithLogitsLoss,
    and FocalLoss. The `loss_func` parameter should be set to one of these base losses.

    The effective number of samples per class is calculated as:

        effective_num = 1 - beta^(samples_per_cls)

    The weights for each class are then calculated as:

        weights = (1 - beta) / effective_num
        weights = weights / sum(weights) * num_classes

    The loss is calculated as:

        loss = (weights * base_loss).mean()

    where `base_loss` is the value returned by the base loss function.

    Args:
        samples_per_cls (list or numpy array): Number of samples per class in the training data.
        beta (float): Degree of balancing between the classes.
        num_classes (int): Number of classes in the classification problem.
        loss_func (nn.Module): Base loss function to use for calculating the loss. Should be one of
            the following: nn.CrossEntropyLoss, nn.BCEWithLogitsLoss, or FocalLoss.

    Returns:
        torch.Tensor: Computed loss value.

    Examples:
        >>> samples_per_cls = [100, 200, 300]
        >>> beta = 0.99
        >>> num_classes = 3
        >>> loss_func = nn.CrossEntropyLoss()
        >>> loss = CB_Loss(samples_per_cls, beta, num_classes, loss_func)
        >>> outputs = torch.randn(4, 3)
        >>> targets = torch.tensor([0, 1, 2, 1])
        >>> output = loss(outputs, targets)

    References:
        - "Class-Balanced Loss Based on Effective Number of Samples"
          https://arxiv.org/abs/1901.05555
    """

    def __init__(self, samples_per_cls, beta, num_classes, loss_func):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_func = loss_func

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        self.weights = (1.0 - beta) / np.array(effective_num)
        self.weights = self.weights / np.sum(self.weights) * num_classes
        self.weights = torch.tensor(self.weights, device=self.device).float()

    def forward(self, logits, target):
        """
        Compute the class-balanced loss for a batch of input logits and corresponding targets.

        Args:
            logits (torch.Tensor): The input tensor of shape (batch_size, num_classes).
            target (torch.Tensor): The target tensor of shape (batch_size,) containing the class
                labels for each input sample.

        Returns:
            The class-balanced loss as a scalar Tensor.
        """
        if self.loss_func.reduction == 'none':
            base_loss = self.loss_func(logits, target)
            weights = self.weights.index_select(0, target)
            balanced_loss = (weights * base_loss).mean()

            return balanced_loss
        else:
            raise ValueError(f"Invalid reduction method: {self.loss_func.reduction}. Please use 'none'.")
