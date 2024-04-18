
"""
PyNets: A Python package for neural network training and evaluation.
Author: George Miloshevich
date: 2024


"""

import logging
import time
import torch
import optuna
import pickle
import warnings


import copy
import os
import json

from torch.utils.data import DataLoader

from datasets import DataFrameDataset
import models

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    

class Trainer:
    """
    A class that represents a trainer for a machine learning model. It is used to train a model using the given data loaders. 

    Parameters
    ----------
    load_data_kwargs : dict, optional
        Keyword arguments for loading the data, by default None
    model_kwargs : dict, optional
        Keyword arguments for initializing the model, by default None
    optimizer_kwargs : dict, optional
        Keyword arguments for initializing the optimizer, by default None
    scheduler_kwargs : dict, optional
        Keyword arguments for initializing the scheduler, by default None
    logger_kwargs : dict, optional
        Keyword arguments for initializing the logger, by default None
    device : str, optional
        The device to use for training, by default None
    work_dir : str, optional
        The directory to save the training results, by default None (no saving)
    
    """
    def __init__(self, dataset_kwargs=None, load_data_kwargs=None, model_kwargs=None, optimizer_kwargs=None, scheduler_kwargs=None, logger_kwargs=None, 
                 device=None,work_dir=None):
        """
        Constructor for the Trainer class.

        If work_dir is provided and there is no configuration file, it will save the configuration file based on the attributes
        of the class. If work_dir is provided and there is a configuration file, it will load the configuration from the file and
        ignore the inputs to the constructor. 
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
                logger.warning(f"Config file {config_file} found. Loading configuration based on it and ignoring the inputs to the constructor!")
                self.__dict__.update(**config) # update the attributes with the configuration file
                dataset_kwargs = config['dataset_kwargs']
                load_data_kwargs = config['load_data_kwargs']
                model_kwargs = config['model_kwargs']
                device = config['device']
                work_dir = config['work_dir']
            else:
                config = copy.deepcopy(self.__dict__) # save the attributes to the config file
                logger.info(f"Creating a new configuration file: {config_file}")
                #print(f"{config =}")
                os.makedirs(os.path.dirname(self.work_dir), exist_ok=True)
                try:
                    with open(config_file, 'w') as f:
                        f.write(json.dumps(config, indent=4))
                except Exception as e:
                    logger.error(f"Error saving configuration file: {e}")
            
        #logging.basicConfig(level=logging.INFO)
        # Create a custom logger
        if work_dir is not None:
            f_handler = logging.FileHandler(f'{self.work_dir}/training.log')
            f_handler.setLevel(logging.DEBUG)
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(f_format)
            logger.addHandler(f_handler)
            logger.info(f"Logging to {self.work_dir}/training.log")
        self.config = copy.deepcopy(self.__dict__) # save the attributes to the config of the trainer class
        self.dataset_kwargs.pop('samples_file',None)  # guardrails against accidentally passing samples_file to DataFrameDataset     
        train_sample = self.dataset_kwargs.pop('train_sample')
        val_sample = self.dataset_kwargs.pop('val_sample')
        test_sample = self.dataset_kwargs.pop('test_sample')
        self.train_dataset = DataFrameDataset(samples_file=train_sample,norm_folder=self.work_dir,**self.dataset_kwargs)
        self.dataset_kwargs.pop('scaler_features', None) # removing these to avoid passing them to the validation and test datasets
        self.dataset_kwargs.pop('scaler_targets', None)
        self.val_dataset = DataFrameDataset(samples_file=val_sample,norm_folder=self.work_dir, 
                                            scaler_features=(self.train_dataset.features_mean,self.train_dataset.features_std), 
                                            scaler_targets=(self.train_dataset.targets_mean,self.train_dataset.targets_std),
                                            **self.dataset_kwargs)
        self.test_dataset = DataFrameDataset(samples_file=test_sample,norm_folder=self.work_dir,
                                            scaler_features=(self.train_dataset.features_mean,self.train_dataset.features_std),
                                            scaler_targets=(self.train_dataset.targets_mean,self.train_dataset.targets_std),
                                             **self.dataset_kwargs)
        
        self.comprehend_config()
    
    def comprehend_config(self):
        """
        Comprehends the configuration settings for the model, criterion, optimizer, and scheduler. The method deep copies
        the configuration settings and extracts the necessary parameters for the model. At the end it calls the method
        trainer::load_data to create/update the data loaders to its value.

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
            logger.warning(f"Creating new model with {model_kwargs =}. Note this will replace any previous model")
            model_name = model_kwargs.pop('model')
            model_class = getattr(models, model_name)
            self.model = model_class(**model_kwargs)
            logger.info(f"Successfully parsed the {model_name} class")
            logger.info(f"Creating object: {self.model}")
        else: # if model is already instantiated
            logger.info(f"Model provided as an input {model_kwargs =}")
            self.model = model_kwargs
        self.device = self._get_device(device)
        self.model.to(self.device)
        logger.info(f"{self.model.device = }")
        
       

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
        self.train_loader = DataLoader(self.train_dataset, **load_data_kwargs['train_loader_kwargs'])
        self.val_loader = DataLoader(self.val_dataset, **load_data_kwargs['val_loader_kwargs'])

    def load_run(self, run):
        """
        Load the configuration file of a specific run, comprehend config the associated file associated 
        with the model and load model weights.
        """
        config_file = os.path.join(self.work_dir, run, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.warning(f"Config file {config_file} found. Loading configuration based on it and the associated model weights/training loss!")
            if config['dataset_kwargs'] != self.config['dataset_kwargs']:
                raise ValueError("The old and new config file have dataset_kwargs which are not consistent! You must create a new run")
            self.config  = config
            self.comprehend_config()
            model_file = f"{self.config['work_dir']}/{run}/model.pth"
            loss_file = f"{self.config['work_dir']}/{run}/loss_dict.pkl"
            logger.info(f"Loading model weights from {model_file}")
            self.model.load_state_dict(torch.load(model_file))
            with open(loss_file, 'rb') as f:
                logger.info(f"Loading loss dictionary from {loss_file}")
                loss_dict = pickle.load(f)
            self.train_loss_, self.val_loss_ = loss_dict['train_loss'], loss_dict['val_loss']
        else:
            raise FileNotFoundError(f"Config file {config_file} not found.")
        
    def fit(self, config=None):
        """
        Performs the training process for the model. 
        It iterates over the specified number of epochs and performs the training and validation steps. 
        The best model weights are saved corresponding to the epoch with the best validation loss.

        Returns:
            None
        """
        trial = None
        if config is not None:
            new_config = copy.deepcopy(config) # to avoid changing the original dictionary
            trial = new_config.pop('trial',None) # handling optuna trials

            for key in new_config['dataset_kwargs']:
                if key in new_config['dataset_kwargs'] and new_config['dataset_kwargs'][key] != self.config['dataset_kwargs'][key]:
                    logger.warning(f"{key} is inconsistent between self.config and new_config. Check that you are not mixing test/train/validation")
            logger.warning(f"Updating the config with the new config")
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
                    
        #if isinstance(self.model,list):
        #    best_loss = []
        #    for model in self.model:
        #        best_loss.append(model.fit(trial))
        #else:
        best_loss = self.model.fit(self.train_loader, self.val_loader, trial=trial)

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

