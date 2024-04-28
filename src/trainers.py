
"""
PyNets: A Python package for neural network training and evaluation.
Author: George Miloshevich
date: 2024


"""

import logging
import torch
import pickle
import warnings


import copy
import os
import json

from torch.utils.data import DataLoader

from . import datasets
from . import models
from . import utilities as ut

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    

class Trainer:
    """
    A class that represents a trainer object for training machine learning models.

    Args:
        dataset_kwargs (dict, optional): Keyword arguments for creating dataset object.
        load_data_kwargs (dict, optional): Keyword arguments for data loaders.
        model_kwargs (dict, optional): Keyword arguments for creating the model.
        device (str, optional): Device to use for training. Defaults to None.
        work_dir (str, optional): Directory to save training outputs. Defaults to None.

    Attributes:
        work_dir (str): Directory to save training outputs.
        dataset_kwargs (dict): Keyword arguments for creating dataset object.
        load_data_kwargs (dict): Keyword arguments for data loaders.
        model_kwargs (dict): Keyword arguments for creating the model.
        device (str): Device to use for training.
        config (dict): A copy of the attributes of the Trainer object.
        train_dataset (DataFrameDataset): Training dataset.
        val_dataset (DataFrameDataset): Validation dataset.
        test_dataset (DataFrameDataset): Test dataset.
        train_loader (DataLoader): Data loader for training dataset.
        val_loader (DataLoader): Data loader for validation dataset.
        model (nn.Module): Machine learning model.

    Methods:
        __init__(self, dataset_kwargs=None, load_data_kwargs=None, model_kwargs=None, device=None, work_dir=None):
            Initializes a Trainer object.
        comprehend_config(self):
            Comprehends the configuration settings for the model and its configs.
        load_data(self, load_data_kwargs):
            Creates the data loaders for training/validation and testing.
        load_run(self, run):
            Load the configuration file of a specific run, comprehend config the associated file associated with the model and load model weights.
        fit(self, config=None):
            Fits the model to the training data and returns the best loss.

    """
    def __init__(self, dataset_kwargs=None, load_data_kwargs=None, model_kwargs=None, 
                 device=None,work_dir=None):
        """
        Initialize a Trainer object.

        Args:
            dataset_kwargs (dict, optional): Keyword arguments for creating dataset object.
            load_data_kwargs (dict, optional): Keyword arguments for data loaders.
            model_kwargs (dict, optional): Keyword arguments for creating the model
            device (str, optional): Device to use for training. Defaults to None.
            work_dir (str, optional): Directory to save training outputs. Defaults to None.
        """
        self.work_dir = work_dir
        self.dataset_kwargs = dataset_kwargs
        self.load_data_kwargs = load_data_kwargs
        self.model_kwargs = model_kwargs
        self.device = device
        
        # === Deal with the configuration file === #
        if work_dir is not None:
            
            os.makedirs(os.path.dirname(self.work_dir), exist_ok=True)
            config_file = os.path.join(self.work_dir, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.warning(f"Config file {config_file} found -> loading configuration")
                self.__dict__.update(**config) # update the attributes with the configuration file
                #dataset_kwargs = config['dataset_kwargs']
                #load_data_kwargs = config['load_data_kwargs']
                #model_kwargs = config['model_kwargs']
                #device = config['device']
                #work_dir = config['work_dir']
            else:
                config = copy.deepcopy(self.__dict__) # save the attributes to the config file
                logger.info(f"Creating a new configuration file: {config_file}")
                os.makedirs(os.path.dirname(self.work_dir), exist_ok=True)
                try:
                    with open(config_file, 'w') as f:
                        f.write(json.dumps(config, indent=4))
                except Exception as e:
                    logger.error(f"Error saving configuration file: {e}")
            
        if self.work_dir is not None:
            f_handler = logging.FileHandler(f'{self.work_dir}/training.log')
            f_handler.setLevel(logging.DEBUG)
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(f_format)
            logger.addHandler(f_handler)
            logger.info(f"========Logging to {self.work_dir}/training.log===========") 
            # Define the extra loggers and add the same FileHandler to them
            datasets_logger = logging.getLogger(__name__)
            datasets_logger.addHandler(f_handler)
            models_logger = logging.getLogger(models.__name__)
            models_logger.addHandler(f_handler)
            read_pic_logger = logging.getLogger(datasets.__name__)
            read_pic_logger.addHandler(f_handler)

        self.config = copy.deepcopy(self.__dict__) # save the attributes to the config of the trainer class
        self.dataset_kwargs.pop('samples_file',None)  # guardrails against accidentally passing samples_file to DataFrameDataset     
        train_sample = self.dataset_kwargs.pop('train_sample')
        val_sample = self.dataset_kwargs.pop('val_sample')
        test_sample = self.dataset_kwargs.pop('test_sample')
        self.train_dataset = datasets.DataFrameDataset(samples_file=train_sample,norm_folder=self.work_dir,**self.dataset_kwargs)
        self.dataset_kwargs.pop('scaler_features', None) # removing these to avoid passing them to the validation and test datasets
        self.dataset_kwargs.pop('scaler_targets', None)
        self.val_dataset = datasets.DataFrameDataset(samples_file=val_sample,norm_folder=self.work_dir, 
                                            scaler_features=(self.train_dataset.features_mean,self.train_dataset.features_std), 
                                            scaler_targets=(self.train_dataset.targets_mean,self.train_dataset.targets_std),
                                            **self.dataset_kwargs)
        self.test_dataset = datasets.DataFrameDataset(samples_file=test_sample,norm_folder=self.work_dir,
                                            scaler_features=(self.train_dataset.features_mean,self.train_dataset.features_std),
                                            scaler_targets=(self.train_dataset.targets_mean,self.train_dataset.targets_std),
                                             **self.dataset_kwargs)
        
        self.comprehend_config()
    
    def comprehend_config(self):
        """
        Comprehends the configuration settings for the model and its configs. The method deep copies
        the configuration settings and extracts the necessary parameters for the model. At the end it calls the method
        trainer::load_data to create/update the data loaders accordingly.

        Returns:
            None
        """
        # === Deal with the model === #
        config = copy.deepcopy(self.config)
        #dataset_kwargs = config['dataset_kwargs']
        load_data_kwargs = config['load_data_kwargs']
        model_kwargs = config['model_kwargs']
        device = config['device']
        if 'run' in config:  # this is to handle optuna trials or other runs
            self.run = config['run']
        else:
            self.run = ''

        if isinstance(model_kwargs, dict): # if model is not instantiated we create it
            logger.warning(f"Creating new model. Note this will replace any previous model")
            model_name = model_kwargs.pop('model')
            model_class = getattr(models, model_name)
            self.model = model_class(**model_kwargs) # < ======= TODO: Add multiple models here
            logger.info(f"Successfully parsed the {model_name} class")
            logger.info(f"Creating object: {self.model}")
        else: # if model is already instantiated
            logger.info(f"Model provided as an input {model_kwargs =}")
            self.model = model_kwargs
        self.device = self._get_device(device)
        self.model.to(self.device)
        logger.info(f"{self.model.device = }")
        
        logger.info(f"Code version git hash: {ut.get_git_revision_hash()}") # TODO: add this to the config file and raise error if it doesn't coincide wiht the current version

        self.load_data(load_data_kwargs)
    
    def load_data(self, load_data_kwargs):
        """
        Creates the data loaders for training/validation and testing.

        Args:
            load_data_kwargs (dict): A dictionary containing the following keyword arguments:
                - train_loader_kwargs (dict): Keyword arguments for configuring the train data loader.
                - val_loader_kwargs (dict): Keyword arguments for configuring the validation data loader.

        Returns:
            None
        """
        self.train_loader = datasets.ChannelDataLoader(self.train_dataset, **load_data_kwargs['train_loader_kwargs'])
        self.val_loader = datasets.ChannelDataLoader(self.val_dataset, **load_data_kwargs['val_loader_kwargs'])

    def load_run(self, run):
        """
        Load the configuration file of a specific run, comprehend config the associated file associated 
        with the model and load model weights.
        """
        config_file = os.path.join(self.work_dir, run, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"==========Config file {config_file} found========")
            logger.warning(f"Loading configuration based on it and the associated model weights/training loss!")
            if config['dataset_kwargs'] != self.config['dataset_kwargs']:
                raise ValueError("The old and new config file have dataset_kwargs which are not consistent! You must create a new run")
            self.config  = config
            self.comprehend_config()
            model_file = f"{self.config['work_dir']}/{run}/model.pth"    # < ======= TODO: Add multiple models here
            loss_file = f"{self.config['work_dir']}/{run}/loss_dict.pkl"
            logger.info(f"Loading model weights from {model_file}")
            self.model.load_state_dict(torch.load(model_file))
            with open(loss_file, 'rb') as f:
                logger.info(f"Loading loss dictionary from {loss_file}")
                loss_dict = pickle.load(f)
            self.model.train_loss_, self.model.val_loss_ = loss_dict['train_loss'], loss_dict['val_loss']
        else:
            raise FileNotFoundError(f"Config file {config_file} not found.")
        
    def fit(self, config=None):
        """
        Fits the model to the training data and returns the best loss.

        Args:
            config (dict): A dictionary containing the configuration parameters for the training process. 
            If provided, the original configuration will be updated with the new configuration.

        Returns:
            float: The best loss achieved during the training process.

        Raises:
            FileExistsError: If the config file already exists and overwriting is not allowed.
            Exception: If there is an error saving the configuration file.

        Notes:
            - If `config` is provided, the original configuration will be updated with the new configuration before training.
            - If `self.run` is not empty, the new configuration will be saved to a subdirectory.
            - The model weights and loss history will be saved to `self.work_dir` if it is not None.
        """
        trial = None
        if config is not None:
            new_config = copy.deepcopy(config) # to avoid changing the original dictionary
            trial = new_config.pop('trial',None) # handling optuna trials

            for key in new_config['dataset_kwargs']:
                if key in new_config['dataset_kwargs'] and new_config['dataset_kwargs'][key] != self.config['dataset_kwargs'][key]:
                    logger.warning(f"{key} is inconsistent between self.config and new_config. Check that you are not mixing test/train/validation")
            logger.warning(f"============Updating the config with the new config============")
            self.config = new_config
            self.comprehend_config()
            if self.run != '': # if we are running a trial, we need to save config to subdirectory
                if os.path.exists(f"{self.work_dir}/{self.run}/config.json"):
                    raise FileExistsError(f"Config file {self.work_dir}/{self.run}/config.json already exists. Overwriting not allowed!")
                logger.info(f"Saving the new configuration to {self.work_dir}/{self.run}/config.json")
                os.makedirs(os.path.dirname(f"{self.work_dir}/{self.run}/"), exist_ok=True)
                config_file = os.path.join(f"{self.work_dir}/{self.run}/", 'config.json')
                try:
                    with open(config_file, 'w') as f:
                        f.write(json.dumps(self.config, indent=4))
                except Exception as e:
                    logger.error(f"Error saving configuration file: {e}")

        best_loss = self.model.fit(self.train_loader, self.val_loader, trial=trial)   # < ======= TODO: Add multiple models here

        if self.work_dir is not None:
            logger.info(f"Saving the model weights and loss history to {self.work_dir}/{self.run}/")
            os.makedirs(os.path.dirname(f"{self.work_dir}/{self.run}/"), exist_ok=True)
            model_path = f"{self.work_dir}/{self.run}/model.pth"
            torch.save(self.model.state_dict(), model_path)
            loss_path = f"{self.work_dir}/{self.run}/loss_dict.pkl"
            with open(loss_path, 'wb') as f:
                pickle.dump( {'train_loss': self.model.train_loss_,'val_loss': self.model.val_loss_}, f)
       
        return best_loss

    def _get_device(self, device):
        """
        Get the device to use for training.
        """
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev

