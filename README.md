
<div align="center">
  <img src="cbloss.svg" alt="PyPI" >
  
</div>

<div align="center">
  <img src="https://img.shields.io/pypi/v/cbloss" alt="PyPI" >
  <img src="https://github.com/wildoctopus/cbloss/actions/workflows/ci.yml/badge.svg" alt="Build Status">
</div>

# cbloss (Class Balanced Loss)
`cbloss` is a Python package that provides Pytorch implementation of - !["Class-Balanced Loss Based on Effective Number of Samples"](https://arxiv.org/abs/1901.05555).

This package also includes Pytorch Implementation of ![Focal loss for dense object detection](https://arxiv.org/pdf/1708.02002.pdf) as `Focal Loss` is currently not avaialable in `torch.nn` module.



## Installation

You can install `cbloss` via pip:

```
pip install cbloss
```



## Usage

### Focal Loss

Focal Loss is a popular loss function for imbalanced classification problems. It's a modification of the standard Cross Entropy nad Binary Classification Loss, that is designed to address class imbalance issues. In essence, it gives more weight to hard to classify examples.
```
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
    gamma = focusing parameter, default to 2, the degree of focusing, 0 means no focusing.
    
Args:
    num_classes (int) : number of classes
    alpha (float): balancing parameter, default to 1.
    gamma (float): focusing parameter, default to 2.
    reduction (str): reduction method for the loss, either 'none', 'mean' or 'sum', default to 'mean'.
```
```python
from cbloss.loss import FocalLoss

loss_fn = FocalLoss(num_classes=3, gamma=2.0, alpha=0.25)
```

### ClassBalancedLoss

This loss function helps address the problem of class imbalance in the training data by assigning higher weights to underrepresented classes during training. The weights are determined based on the `number of samples per class and a beta value`, which controls the degree of balancing between the classes.

The loss function supports different types of base losses, including `CrossEntropyLoss`, `BCEWithLogitsLoss`, and `FocalLoss`.
```
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
            the following: nn.CrossEntropyLoss, nn.BCEWithLogitsLoss, or FocalLoss..
```
```python
from cbloss.loss import ClassBalancedLoss

samples_per_cls = [300, 200, 100] # an example case
loss_func = nnCrossEntropyLoss(reduction = 'none')

loss_fn = ClassBalancedLoss(samples_per_cls, beta=0.99, num_classes=3, loss_func=loss_func)

```
The `loss_func` parameter should be set to one of these base losses (FocalLoss, nn.CrossEntropyLoss, nn.BCEWithLogitsLoss). 

`*** Please Note "reduction = 'none'" should be set for all base Loss Function, while using ClassBalancedLoss.` 


## v0.1.0

If you have v0.1.0 installed, please use cb_loss.loss to import FocalLoss and ClassBalancedLoss.

```python
from cbloss.loss import ClassBalancedLoss

samples_per_cls = [300, 200, 100] # an example case
loss_func = nnCrossEntropyLoss(reduction = 'none')

loss_fn = ClassBalancedLoss(samples_per_cls, beta=0.99, num_classes=3, loss_func=loss_func)

```


# Citations
```
      @inproceedings{lin2017focal,
        title={Focal Loss for Dense Object Detection},
        author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Dollar, Piotr},
        booktitle={Proceedings of the IEEE international conference on computer vision},
        pages={2980--2988},
        year={2017}
      }

      @inproceedings{cui2019class,
        title={Class-balanced loss based on effective number of samples},
        author={Cui, Yifan and Jia, Meng and Lin, Tsung-Yi and Song, Yang and Belongie, Serge},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={9268--9277},
        year={2019}
      }
```


# Contribution and Support


Contributions, issues, and feature requests are welcome! Feel free to check out the ![issues page](https://github.com/wildoctopus/cbloss/issues) if you want to contribute.

If you find any bugs or have any questions, please open an issue on the repository or contact me through the email listed in our profiles.

If you find this project helpful, please give a ⭐️ on GitHub and share it with your friends and colleagues. This will help me grow and improve the project. Thank you for your support!


