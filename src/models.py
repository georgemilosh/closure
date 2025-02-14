
"""
PyNets: A Python package for neural network training and evaluation.
Author: George Miloshevich
date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#import optuna
import time
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import sys

import logging
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
#optuna.logging.get_logger(__name__).addHandler(logging.StreamHandler(sys.stdout))
#optuna.logging.enable_propagation()  # Propagate logs to the root logger.

class PyNet:
    """
    base class for managing neural nets
    Args:
    
    model_seed : int, optional
        Seed for the model weights, by default None
    optimizer_kwargs : dict, optional
        Keyword arguments for initializing the optimizer, by default None
    scheduler_kwargs : dict, optional
        Keyword arguments for initializing the scheduler, by default None
    logger_kwargs : dict, optional
        Keyword arguments for initializing the logger, by default None

    Attributes:
    model : (torch.nn.Module) : The neural network model.
    """
    def __init__(self, model='FCNN', model_seed=None, optimizer_kwargs=None, scheduler_kwargs=None, logger_kwargs=None,
                rank=None, local_rank=None, **kwargs): 
        model_class =  globals() [model] #getattr(__name__, model)
        self.local_rank = local_rank
        self.rank = rank
        logger.info(f"{torch.cuda.is_available() = }")
        self.model_ = model_class(**kwargs).to(self.local_rank)
        logger.info(f"Initializing model {self.model_}")
        try:
            self.model = DDP(self.model_, device_ids=[self.local_rank])
        except Exception:
            self.model = self.model_
            logger.info(f"DDP not available, initializing model using single GPU {self.local_rank = }")
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
        self.save_every = self.scheduler_kwargs.pop('save_every', None)
        self.logger_kwargs = logger_kwargs
        

        if model_seed is not None:
            torch.manual_seed(model_seed) # set the seed for the weights
        self.define_optimizer_sheduler()

    def define_optimizer_sheduler(self):
        """
        Defines the optimization criterion, optimizer, and scheduler for the model. 

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
            self.optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters(), **self.optimizer_kwargs)
        else:
            self.optimizer = optimizer_name # assuming that optimizer is already passed, e.g. optimizer = torch.optim.Adam(model.parameters())
        
        # === Deal with the scheduler === #
        scheduler_name = self.scheduler_kwargs.pop('scheduler')
        if scheduler_name is not None:
            self.scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(self.optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None

    @property
    def device(self):
        """
        Get the device of the model.
        """
        return next(self.model.parameters()).device
    @property
    def total_parameters(self):
        """
        Get the parameters of the model.
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def predict(self, features):
        """
        Predict the output using the given some input feature array (can be numpy).
        Also sets the dtype of the features and targets to be used in the predict method.

        Returns:
            torch.Tensor: The predicted output tensor.
        """
        self.model.eval()
        predictions = []
        loader = DataLoader(features, batch_size=32)
        for features in loader:
            with torch.no_grad():
                out = self.model(features.to(self.device))
            predictions.append(out)
        self.features_dtype = features.dtype
        self.targets_dtype = out.dtype
        return torch.cat(predictions)

    def _compute_loss(self, prediction, ground_truth, criterion):
        """
        Compute the loss between the ground_truth and prediction values using the provided criterion.        
        """
        if not isinstance(prediction, torch.Tensor):
            logger.warning(f'Object target is not a tensor. Casting to tensor of {self.targets_dtype}.')
            prediction = torch.tensor(prediction, dtype=self.targets_dtype)
        if not isinstance(ground_truth, torch.Tensor):
            logger.warning(f'Object ground_truth is not a tensor. Casting to tensor of {self.targets_dtype}.')
            ground_truth = torch.tensor(ground_truth, dtype=self.targets_dtype)
        try:
            loss = criterion(prediction.to(self.device), ground_truth.to(self.device))
        except Exception as e:
            logger.info(f"{ground_truth.shape =  }, {prediction.shape = }, {criterion = }, {prediction.dtype = }, {ground_truth.dtype = }")
            logger.error(f"Error in computing loss: {e}")
            raise e

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
            self.model.train()
        else:
            self.model.eval()
        num_batches = len(loader)
        if self.metrics is not None:
            running_metrics = {metric._get_name(): 0.0 for metric in self.metrics}
        else:
            running_metrics = {}
        running_metrics['criterion'] = 0.0
        for batch_idx, (features, targets) in enumerate(loader):
            features, targets = self._to_device(features, targets, self.local_rank)
            self.optimizer.zero_grad() # zero the parameter gradients
            with torch.set_grad_enabled(phase == 'train'): # track gradients only if in train
                out = self.model(features)
                try:
                    loss = self._compute_loss(out, targets, self.criterion)
                except Exception as e:
                    logger.error(f"{out.shape = }, {targets.shape = }, {features.shape = }, {self.criterion = }")
                    raise e
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
            running_metrics['criterion'] += loss.item()
            if self.metrics is not None:
                for metric in self.metrics:
                    running_metrics[metric._get_name()] += self._compute_loss(out, targets, metric).item()
        if phase == 'train':
            self.train_batch_idx = batch_idx
        else:
            self.val_batch_idx = batch_idx

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
        if not hasattr(self, 'train_loss_'):
            self.train_loss_ = {} # training history
        if not hasattr(self, 'val_loss_'):
            self.val_loss_ = {} # validation history

        total_start_time = time.time() # track total training time
        best_loss = torch.inf  # track best loss
        epoch_best = 0
        self.lr = []
        num_batches_train = len(train_loader)
        logger.info(f"Each forward pass had {num_batches_train} train batches.")
        logger.info(f"Number of samples per batch: {len(next(iter(train_loader))[0]) = }")
        self.total_time = {"total": None, "train+val" : [], "train" : []}
        # ---- train process ----
        for epoch in range(self.epochs):
            epoch_start_time = time.time() # track epoch time
            tr_loss = self._forward_pass(train_loader, phase='train')
            epoch_time_train = time.time() - epoch_start_time
            self.total_time["train"].append(epoch_time_train)
            #if self.rank == 0:
            val_loss = self._forward_pass(val_loader, phase='val')
            if self.scheduler is not None:
                self.scheduler.step(val_loss['criterion'])

            for key in tr_loss:
                if key not in self.train_loss_:
                    self.train_loss_[key] = []
                    self.val_loss_[key] = []
                self.train_loss_[key].append(tr_loss[key])
                #if self.rank == 0:
                self.val_loss_[key].append(val_loss[key])
            
            epoch_time = time.time() - epoch_start_time
            self.total_time["train+val"].append(epoch_time)
            if self.rank == 0:
                if val_loss["criterion"] < best_loss:
                    if self.save_every == 'best' or (isinstance(self.save_every, int) and epoch % self.save_every == 0):
                        best_loss = val_loss["criterion"]
                        best_weights = copy.deepcopy(self.model.state_dict()) # note that this operation may take some time
                        epoch_best = epoch
                    #torch.save(self.model.state_dict(), self.work_dir) # this saves every epoch if improvement
                self._logging(tr_loss, val_loss, epoch+1,self.epochs, epoch_time, epoch_time_train, **self.logger_kwargs)

                if self.scheduler is not None:
                    try:
                        self.lr.append(self.scheduler.get_last_lr())
                    except AttributeError: # Compatability with an earlier version of Pytorch
                        self.lr.append(self.scheduler._last_lr)
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
                        logger.info("Raising TrialPruned exception.")
                        #raise optuna.exceptions.TrialPruned()
            else:
                self._logging(tr_loss, None, epoch+1,self.epochs, epoch_time, epoch_time_train, **self.logger_kwargs)
                
        if self.rank == 0:
            if self.save_every is not None:
                # restore model and return best accuracy
                logger.info(f"Best loss: {best_loss} at epoch {epoch_best+1}, restoring the corresponding weights...")
                self.model.load_state_dict(best_weights)

        total_time = time.time() - total_start_time
        self.total_time["total"] = total_time
        logger.info(f"Each forward pass had {num_batches_train} train batches and forward pass had final {self.train_batch_idx = }.")
        logger.info(f"Number of samples per batch: {len(next(iter(train_loader))[0]) = }")
        # final message
        logger.info(f"""End of training on | {self.rank = }, {self.device = }. Total time: {round(total_time, 5)} seconds""")
        return best_loss

        
    def _to_device(self, features, targets, device):
        """
        Move the features and targets to the specified device.
        """
        return features.to(device), targets.to(device)
    
    def _logging(self, tr_loss, val_loss, epoch, epochs, epoch_time, epoch_time_train, show=True, update_step=20):
        """
        Log the training progress. 
        """
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss['criterion']:.4e}" 
                if val_loss is not None and 'criterion' in val_loss and val_loss['criterion'] is not None:
                    msg = f"{msg} | Val loss: {val_loss['criterion']:.4e}"
                msg = f"{msg} | Time/epoch: {round(epoch_time, 3)} s"
                msg = f"{msg} | Time/epoch_train: {round(epoch_time_train, 3)} s"
                if self.scheduler is not None:
                    try:
                        msg = f"{msg} | Learn rate: {self.scheduler.get_last_lr():.4e}"
                    except AttributeError: # Compatability with an earlier version of Pytorch
                        try:
                            msg = f"{msg} | Learn rate: {self.scheduler._last_lr:.4e}"
                        except TypeError: 
                            msg = f"{msg} | Learn rate: {self.scheduler._last_lr}"

                logger.info(msg)

class CNet(torch.nn.Module):
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
    
        #super().define_optimizer_sheduler() # To define optimizer we have to have the layers already defined

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

class ResNet(torch.nn.Module):
    """
    ResNet model for image classification.
    This class represents a ResNet model for image classification. It inherits from the `PyNet` class and implements the forward pass method.
    Attributes:
        channels (list): List of integers representing the number of input and output channels for each convolutional layer.
        kernels (list): List of integers representing the kernel size for each convolutional layer.
        activations (nn.ModuleList): ModuleList containing the activation functions for each convolutional layer.
        batch_norms (nn.ModuleList): ModuleList containing the batch normalization layers for each convolutional layer.
        dropouts (nn.ModuleList): ModuleList containing the dropout layers for each convolutional layer.
        convs (nn.ModuleList): ModuleList containing the convolutional layers.
        skip_connect (dict): Dictionary containing the indices of the layers to be skipped in the skip connection. 
            The key is the index of the current layer, and the value is the index of the layer whose values are added to the current layer before the activation
    Methods:
        __init__(self, channels, kernels, activations=None, batch_norms=None, dropouts=None, skip_connect=None, **kwargs):
            Initializes the ResNet model with the given parameters.
        forward(self, x):
            Performs the forward pass of the ResNet model.
    """    
    def __init__(self, channels, kernels, activations=None, batch_norms=None, 
                 dropouts=None, skip_connect=None, **kwargs):
        super().__init__(**kwargs) # To pass optimizer_kwargs, scheduler_kwargs, logger_kwargs to PyNet constructor
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

        #super().define_optimizer_sheduler() # To define optimizer we have to have the layers already defined

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
    def __init__(self, channels, kernels, activations=None, batch_norms=None, dropouts=None, **kwargs):
        super().__init__(**kwargs) # To pass optimizer_kwargs, scheduler_kwargs, logger_kwargs to PyNet constructor
        seq_list = []
        if activations is None:
            activations = [None] * (len(channels) - 1)
        if batch_norms is None:
            batch_norms = [None] * (len(channels) - 1)
        if dropouts is None:
            dropouts = [None] * (len(channels) - 1)
        for i in range(len(channels)-1):
            seq_list.append(nn.Conv2d(channels[i], channels[i+1], kernels[i], padding=(kernels[i]-1)//2))
            if activations[i] is not None:
                seq_list.append(getattr(nn, activations[i])())
            if batch_norms[i] is not None and batch_norms[i]:
                seq_list.append(nn.BatchNorm2d(channels[i+1]))
            if dropouts[i] is not None and dropouts[i] > 0.0:
                seq_list.append(nn.Dropout2d(dropouts[i]))
        self.seq_model = torch.nn.Sequential(*seq_list)

        #super().define_optimizer_sheduler() # To define optimizer we have to have the layers already defined
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

        #super().define_optimizer_sheduler() # To define optimizer we have to have the layers already defined
    

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