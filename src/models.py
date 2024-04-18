
"""
PyNets: A Python package for neural network training and evaluation.
Author: George Miloshevich
date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import logging


class PyNet(torch.nn.Module):
    """
    base class for neural nets
    """
    def __init__(self, model_seed=None):
        super().__init__() 
        if model_seed is not None:
            torch.manual_seed(model_seed)

    @property
    def device(self):
        """
        Get the device of the model.
        """
        return next(self.parameters()).device
    @property
    def total_parameters(self):
        """
        Get the parameters of the model.
        """
        return sum(p.numel() for p in self.parameters())
    
    


class CNet(PyNet):
    def __init__(self):
        super().__init__() 
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.conv3 = nn.Conv2d(16, 32, 5) 
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(32 * 28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.pool(F.relu(self.conv4(x))) 
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
   

class MLP(PyNet):
    """
    Multi-Layer Perceptron (MLP) model.

    Args:
        feature_dims (list): A list of integers representing the dimensions of the input and output features.

    Attributes:
        flatten (torch.nn.Flatten): A flatten layer to convert the input tensor into a 1-dimensional tensor.
        linear_relu_stack (torch.nn.Sequential): A sequential container for the linear and ReLU layers.

    Examples:
    --------
    >>> model = MLP([8, 24, 12, 6, 1], activations=['ReLU', 'ReLU', 'ReLU', None])
    >>> print(model)
    MLP(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=8, out_features=24, bias=True)
        (1): ReLU()
        (2): Linear(in_features=24, out_features=12, bias=True)
        (3): ReLU()
        (4): Linear(in_features=12, out_features=6, bias=True)
        (5): ReLU()
        (6): Linear(in_features=6, out_features=1, bias=True)
      )
    )
    
    """
    def __init__(self, feature_dims, activations=None, dropouts=None, **kwargs):
        super().__init__(**kwargs)
        self.flatten = torch.nn.Flatten()
        seq_list = []
        if activations is None:
            activations = [None] * (len(feature_dims) - 1)
        if dropouts is None:
            dropouts = [None] * (len(feature_dims) - 1)
        for i in range(len(feature_dims) - 1):
            linear_layer = torch.nn.Linear(feature_dims[i], feature_dims[i + 1])
            seq_list.append(linear_layer)
            if activations[i] is not None:
                activation_layer = getattr(torch.nn, activations[i])() #  'ReLu' => torch.nn.ReLU() 
                seq_list.append(activation_layer)
            if dropouts is not None:
                if dropouts[i] is not None:
                    dropout_layer = torch.nn.Dropout(dropouts[i])
                    seq_list.append(dropout_layer)
            
           
        self.linear_relu_stack = torch.nn.Sequential(*seq_list[:-1])
    

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out