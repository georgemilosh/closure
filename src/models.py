
"""
PyNets: A Python package for neural network training and evaluation.
Author: George Miloshevich
date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import time
from torch.utils.data import DataLoader
import copy

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyNet(torch.nn.Module):
    """
    base class for neural nets
    Args:
    model_seed : int, optional
        Seed for the model weights, by default None
    optimizer_kwargs : dict, optional
        Keyword arguments for initializing the optimizer, by default None
    scheduler_kwargs : dict, optional
        Keyword arguments for initializing the scheduler, by default None
    logger_kwargs : dict, optional
        Keyword arguments for initializing the logger, by default None
    """
    def __init__(self, model_seed=None, optimizer_kwargs=None, scheduler_kwargs=None, logger_kwargs=None):
        super().__init__() 
        logger.info(f"Initializing {self.__class__.__name__} model.")
        #logger.info(f"{model_seed=}, {optimizer_kwargs=}, {scheduler_kwargs=}, {logger_kwargs=}")
        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = optimizer_kwargs
        if scheduler_kwargs is None:
            self.scheduler_kwargs = {}
        else:
            self.scheduler_kwargs = scheduler_kwargs
        self.early_stopping = self.scheduler_kwargs.pop('early_stopping', None)
        self.epochs = self.scheduler_kwargs.pop('epochs')
        self.logger_kwargs = logger_kwargs
        if model_seed is not None:
            torch.manual_seed(model_seed) # set the seed for the weights

    def define_optimizer_sheduler(self):
        """
        Defines the optimization criterion, optimizer, and scheduler for the model. This function should be called
        by any subclass of PyNet in the __init__ method after the layers have been defined.

        This method handles the following steps:
        1. Sets the optimization criterion based on the provided criterion name.
        2. Sets the metrics to track during optimization.
        3. Sets the optimizer based on the provided optimizer name and parameters.
        4. Sets the scheduler based on the provided scheduler name and parameters.

        Raises:
            NotImplementedError: If the provided scheduler name is not recognized.

        """
        # === Deal with the criterion #
        criterion = self.optimizer_kwargs.pop('criterion')
        try:
            criterion = getattr(torch.nn, criterion)()
            self.criterion = criterion
        except Exception as e:
            logger.error(f"Criterion {criterion} not recognized. Please use a valid criterion from torch.nn.")
            raise e
        logger.info(f"Optimization criterion {self.criterion}")

        # === Deal with the optimizer === #
        metrics = self.optimizer_kwargs.pop('metrics', None)
        self.metrics = metrics
        if metrics is not None:
            self.metrics = [getattr(torch.nn, metric)() for metric in metrics]
            logger.info(f"Tracking metrics {self.metrics}")
        
         # === Deal with the optimizer === #
        optimizer_name = self.optimizer_kwargs.pop('optimizer')
        if isinstance(optimizer_name, str):
            self.optimizer = getattr(torch.optim, optimizer_name)(self.parameters(), **self.optimizer_kwargs)
        else:
            self.optimizer = optimizer_name # assuming that optimizer is already passed, e.g. optimizer = torch.optim.Adam(model.parameters())
        
        # === Deal with the scheduler === #
        scheduler_name = self.scheduler_kwargs.pop('scheduler')
        self.scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(self.optimizer, **self.scheduler_kwargs)

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
    
    def predict(self, features):
        """
        Predict the output using the given some input feature array (can be numpy).
        Also sets the dtype of the features and targets to be used in the predict method.

        Returns:
            torch.Tensor: The predicted output tensor.
        """
        self.eval()
        predictions = []
        loader = DataLoader(features, batch_size=32)
        for features in loader:
            with torch.no_grad():
                out = self(features.to(self.device))
            predictions.append(out)
        self.features_dtype = features.dtype
        self.targets_dtype = out.dtype
        return torch.cat(predictions)

    def _compute_loss(self, ground_truth, prediction, criterion):
        """
        Compute the loss between the ground_truth and prediction values using the provided criterion.        
        """
        if not isinstance(prediction, torch.Tensor):
            logger.warning(f'Object target is not a tensor. Casting to tensor of {self.targets_dtype}.')
            prediction = torch.tensor(prediction, dtype=self.targets_dtype)
        if not isinstance(ground_truth, torch.Tensor):
            logger.warning(f'Object ground_truth is not a tensor. Casting to tensor of {self.targets_dtype}.')
            ground_truth = torch.tensor(ground_truth, dtype=self.targets_dtype)
        loss = criterion(ground_truth, prediction)

        # apply regularization if any
        # loss += penalty.item() 
        return loss
    
    def _forward_pass(self, loader, phase='test'):
        """
        Perform a forward pass (single epoch) through the model using the given data loader. 

        Args:
            loader (torch.utils.data.DataLoader): The data loader containing the input features and targets.
            phase (str, optional): The phase of the forward pass. Defaults to 'test'. 
            If 'train', the model will be set to training mode and the weights will be updated. 
            If not, the model will be set to evaluation mode and no gradients will be computed.

        Returns:
            float: The average loss over all batches in the data loader at the best epoch.
        """
        if phase == 'train':
            self.train()
        else:
            self.eval()
        num_batches = len(loader)

        if self.metrics is not None:
            running_metrics = {metric._get_name(): 0.0 for metric in self.metrics}
        else:
            running_metrics = {}
        running_metrics['criterion'] = 0.0
        for features, targets in loader:
            features, targets = self._to_device(features, targets, self.device)
            self.optimizer.zero_grad() # zero the parameter gradients
            with torch.set_grad_enabled(phase == 'train'): # track gradients only if in train
                out = self(features)
                loss = self._compute_loss(out, targets, self.criterion)
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
            running_metrics['criterion'] += loss.item()
            if self.metrics is not None:
                for metric in self.metrics:
                    running_metrics[metric._get_name()] += self._compute_loss(out, targets, metric).item()

        for key in running_metrics:
            running_metrics[key] /= num_batches
        return running_metrics
    
    def fit(self, train_loader, val_loader, trial=None):
        """
        Trains the model using the provided training and validation data loaders.

        Args:
            train_loader (DataLoader): The data loader for the training set.
            val_loader (DataLoader): The data loader for the validation set.
            trial (optuna.Trial, optional): An optuna Trial object for hyperparameter optimization. Defaults to None.

        Returns:
            None

        Raises:
            optuna.exceptions.TrialPruned: If the optuna trial is pruned.

        """
        self.train_loss_ = {} # training history
        self.val_loss_ = {} # validation history

        total_start_time = time.time() # track total training time
        best_loss = torch.inf  # track best loss
        epoch_best = 0
        # ---- train process ----
        for epoch in range(self.epochs):
            epoch_start_time = time.time() # track epoch time
            tr_loss = self._forward_pass(train_loader, phase='train')
            
            val_loss = self._forward_pass(val_loader, phase='val')
            if self.scheduler is not None:
                self.scheduler.step(val_loss['criterion'])
                
            for key in tr_loss:
                if key not in self.train_loss_:
                    self.train_loss_[key] = []
                    self.val_loss_[key] = []
                self.train_loss_[key].append(tr_loss[key])
                self.val_loss_[key].append(val_loss[key])
            
            epoch_time = time.time() - epoch_start_time
            
            if val_loss["criterion"] < best_loss:
                best_loss= val_loss["criterion"]
                #torch.save(self.model.state_dict(), self.work_dir) # this saves every epoch if improvement
                best_weights = copy.deepcopy(self.state_dict())
                epoch_best = epoch
            self._logging(tr_loss, val_loss, epoch+1,self.epochs, epoch_time, **self.logger_kwargs)
            
            # ---- early stopping ----
            if self.early_stopping is not None:
                if epoch - epoch_best > self.early_stopping:
                    logger.warning(f"Early stopping engaged at epoch {epoch}")
                    break
            
            # ---- handle optuna ----
            if trial is not None:
                trial.report(val_loss['criterion'], epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
        # restore model and return best accuracy
        logger.info(f"Best loss: {best_loss} at epoch {epoch_best+1}, restoring the corresponding weights...")
        self.load_state_dict(best_weights)

        total_time = time.time() - total_start_time

        # final message
        logger.info(f"""End of training. Total time: {round(total_time, 5)} seconds""")
        return best_loss

        
    def _to_device(self, features, targets, device):
        """
        Move the features and targets to the specified device.
        """
        return features.to(device), targets.to(device)
    
    def _logging(self, tr_loss, val_loss, epoch, epochs, epoch_time, show=True, update_step=20):
        """
        Log the training progress. 
        """
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss}" 
                msg = f"{msg} | Validation loss: {val_loss}"
                msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"
                if self.scheduler is not None:
                    try:
                        msg = f"{msg} | Learning rate: {self.scheduler.get_last_lr()}"
                    except AttributeError: # Compatability with an earlier version of Pytorch
                        msg = f"{msg} | Learning rate: {self.scheduler._last_lr}"

                logger.info(msg)


class CNet(PyNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # To pass optimizer_kwargs, scheduler_kwargs, logger_kwargs to PyNet constructor
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.conv3 = nn.Conv2d(16, 32, 5) 
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(32 * 28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
    
        super().define_optimizer_sheduler() # To define optimizer we have to have the layers already defined

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
    >>> model = MLP([8, 3, 1], weights=[{'name': 'uniform_', 'std' : 1/np.sqrt(8)}, {'name': 'uniform_', 'std' : 1/np.sqrt(3)}, \
        biases = [{'name': 'zeros_'},{'name': 'zeros_'}])]
    
    """
    def __init__(self, feature_dims, activations=None, dropouts=None, weights=None, biases=None, **kwargs):
        super().__init__(**kwargs)

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
        #print(seq_list)
        self.linear_relu_stack = torch.nn.Sequential(*seq_list)

        super().define_optimizer_sheduler() # To define optimizer we have to have the layers already defined
    

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