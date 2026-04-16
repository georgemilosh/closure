
"""
models.py — Neural network architectures for closure.

Pure ``torch.nn.Module`` implementations: CNet, ResNet, FCNN, MLP.

Repo:       closure
Projects:   STRIDE, HELIOSKILL
Author:     George Miloshevich
Date:       2025
License:    MIT License
"""

__all__ = ["CNet", "FCNN", "ResNet", "MLP"]

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class CNet(torch.nn.Module):
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
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet(torch.nn.Module):
    """ResNet-style convolutional model with optional skip connections.

    Parameters
    ----------
    channels : list[int]
        Number of channels for each layer (length N+1 for N conv layers).
    kernels : list[int]
        Kernel size for each conv layer (length N).
    activations : list[str | None] or None
        Activation function names per layer (``torch.nn`` class names).
    batch_norms : list[bool | None] or None
        Whether to apply BatchNorm2d after each layer.
    dropouts : list[float | None] or None
        Dropout2d rate per layer.
    skip_connect : dict[str, int] or None
        Skip connection mapping ``{"target_layer": source_layer}``.
    """
    def __init__(
        self,
        channels: list[int],
        kernels: list[int],
        activations: list[str | None] | None = None,
        batch_norms: list[bool | None] | None = None,
        dropouts: list[float | None] | None = None,
        skip_connect: dict[str, int] | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.kernels = kernels
        if activations is None:
            activations = [None] * (len(channels) - 1)
        if batch_norms is None:
            batch_norms = [None] * (len(channels) - 1)
        if dropouts is None:
            dropouts = [None] * (len(channels) - 1)
        self.skip_connect = {}
        if skip_connect is not None:
            for key, value in skip_connect.items():
                assert int(key) > value, f"Skip connection must be to higher (key) from lower layer (value), but we have {int(key) = }, {value = }"
                self.skip_connect[int(key)] = value
        self.skip_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(len(channels)-1):
            self.convs.append(nn.Conv2d(channels[i], channels[i+1], kernels[i], padding=(kernels[i]-1)//2))
            if i in self.skip_connect:
                self.skip_convs.append(nn.Conv2d(channels[self.skip_connect[i]+1], channels[i+1], kernels[i], padding=(kernels[i]-1)//2))
            else:
                self.skip_convs.append(None)
            if activations[i] is not None:
                if isinstance(activations[i], list):
                    self.activations.append(getattr(nn, activations[i][0])(*activations[i][1:]))
                else:
                    self.activations.append(getattr(nn, activations[i])())
            else:
                self.activations.append(None)
            if batch_norms[i] is not None and batch_norms[i]:
                self.batch_norms.append(nn.BatchNorm2d(channels[i+1]))
            else:
                self.batch_norms.append(None)
            if dropouts[i] is not None and dropouts[i] > 0.0:
                self.dropouts.append(nn.Dropout2d(dropouts[i]))
            else:
                self.dropouts.append(None)

    def forward(self, x):
        """
        Forward pass of the FCNN.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        out = x
        out_store = []
        for i in range(len(self.channels) - 1):
            out = self.convs[i](out)
            if self.skip_convs[i] is not None:
                out += self.skip_convs[i](out_store[self.skip_connect[i]])
                """try:
                    out += self.skip_convs[i](out_store[self.skip_connect[i]])
                except Exception as e:
                    logger.error(f"{i = }, {self.skip_connect[i] = }, {self.skip_convs[i] = }, {self.convs[i] = }, {out_store[self.skip_connect[i]].shape = }")
                    raise e"""
            if self.activations[i] is not None:
                out = self.activations[i](out)
            if self.batch_norms[i] is not None:
                out = self.batch_norms[i](out)
            if self.dropouts[i] is not None:
                out = self.dropouts[i](out)
            out_store.append(out)
        return out
    
class FCNN(torch.nn.Module):
    """Fully convolutional neural network.

    Parameters
    ----------
    channels : list[int]
        Number of channels for each layer (length N+1 for N conv layers).
    kernels : list[int]
        Kernel size for each conv layer (length N).
    activations : list[str | None] or None
        Activation function names per layer.
    batch_norms : list[bool | None] or None
        Whether to apply BatchNorm2d.
    dropouts : list[float | None] or None
        Dropout2d rate per layer.
    """
    def __init__(
        self,
        channels: list[int],
        kernels: list[int],
        activations: list[str | None] | None = None,
        batch_norms: list[bool | None] | None = None,
        dropouts: list[float | None] | None = None,
    ):
        super().__init__()
        seq_list = []
        if activations is None:
            activations = [None] * (len(channels) - 1)
        if batch_norms is None:
            batch_norms = [None] * (len(channels) - 1)
        if dropouts is None:
            dropouts = [None] * (len(channels) - 1)
        for i in range(len(channels)-1):
            # For even kernels, need to add padding or use output_padding in upsampling
            if kernels[i] % 2 == 0:
                padding = kernels[i] // 2
            else:
                padding = (kernels[i] - 1) // 2
            seq_list.append(nn.Conv2d(channels[i], channels[i+1], kernels[i], padding=padding))
            if activations[i] is not None:
                seq_list.append(getattr(nn, activations[i])())
            if batch_norms[i] is not None and batch_norms[i]:
                seq_list.append(nn.BatchNorm2d(channels[i+1]))
            if dropouts[i] is not None and dropouts[i] > 0.0:
                seq_list.append(nn.Dropout2d(dropouts[i]))
        self.seq_model = torch.nn.Sequential(*seq_list)

    def forward(self, x):
        """
        Forward pass of the FCNN.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        out = self.seq_model(x)
        return out
        
class MLP(torch.nn.Module):
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
    >>> model = MLP([8, 3, 1], weights=[{'name': 'uniform_', 'std' : 1/np.sqrt(8)}, {'name': 'uniform_', 'std' : 1/np.sqrt(3)}, \
        biases = [{'name': 'zeros_'},{'name': 'zeros_'}])]
    
    """
    def __init__(
        self,
        feature_dims: list[int],
        activations: list[str | None] | None = None,
        dropouts: list[float | None] | None = None,
        weights: list[dict | None] | None = None,
        biases: list[dict | None] | None = None,
    ):
        super().__init__()

        self.flatten = torch.nn.Flatten()
        seq_list = []
        if activations is None:
            activations = [None] * (len(feature_dims) - 1)
        if dropouts is None:
            dropouts = [None] * (len(feature_dims) - 1)
        if weights is None:
            weights = [None] * (len(feature_dims) - 1)
        if biases is None:
            biases = [None] * (len(feature_dims) - 1)
        for i in range(len(feature_dims) - 1):
            linear_layer = torch.nn.Linear(feature_dims[i], feature_dims[i + 1])
            if weights[i] is not None:
                try:
                    name = weights[i].pop('name')
                except Exception as e:
                    logger.info(f"{weights = }")
                    logger.error(f"Error in weights: {i = }, {weights[i] = }")
                    raise e
                getattr(torch.nn.init, name)(linear_layer.weight, **weights[i])
            if biases[i] is not None:
                name = biases[i].pop('name')
                getattr(torch.nn.init, name)(linear_layer.bias, **biases[i])
            seq_list.append(linear_layer)
            if activations[i] is not None:
                activation_layer = getattr(torch.nn, activations[i])() #  'ReLu' => torch.nn.ReLU() 
                seq_list.append(activation_layer)
            if dropouts is not None:
                if dropouts[i] is not None:
                    dropout_layer = torch.nn.Dropout(dropouts[i])
                    seq_list.append(dropout_layer)
        self.linear_relu_stack = torch.nn.Sequential(*seq_list)

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