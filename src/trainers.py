
"""
PyNets: A Python package for neural network training and evaluation.
Author: George Miloshevich
date: 2024


"""

import logging
import torch
import pickle
import warnings
import psutil
import argparse
import copy
import os
import shutil
import torch.distributed as dist
import json
from socket import gethostname
import csv


from . import datasets
from . import models
from . import utilities as ut

import logging
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

def set_nested_config(config, key, value):
    """
    Set a nested configuration value in a dictionary.
    This function takes a configuration dictionary, a dot-separated key string, 
    and a value. It sets the value in the dictionary at the location specified 
    by the key string, creating nested dictionaries as needed.
    Args:
        config (dict): The configuration dictionary to update.
        key (str): A dot-separated string specifying the nested key.
        value (str): The value to set. It will be converted to an int or float 
                     if possible.
    Example:
        config = {}
        set_nested_config(config, 'a.b.c', '123')
        # config is now {'a': {'b': {'c': 123}}}
    """

    keys = key.split('.')
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Convert value to appropriate type
    if value.isdigit():
        value = int(value)
    else:
        try:
            value = float(value)
        except ValueError:
            pass
    d[keys[-1]] = value

class CustomFilter(logging.Filter):
    """
    A custom filter class for logging that will be used to prepend the rank, local_rank and nodename to the log messages.
    """
    def __init__(self, job_id, rank, local_rank, nodename):
        super().__init__()
        self.job_id = job_id
        self.rank = rank
        self.local_rank = local_rank
        self.nodename = nodename

    def filter(self, record):
        record.job_id = self.job_id
        record.rank = self.rank
        record.local_rank = self.local_rank
        record.nodename = self.nodename
        return True
    

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
        mode_test (bool): whether trainer is intended to not load data.
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
    def __init__(self, dataset_kwargs=None, load_data_kwargs=None, model_kwargs=None, mode_test=False,
                 device=None, work_dir=None, log_name=None, log_level=None, force=False, timing_name=None,
                 world_size=None, rank=None, gpus_per_node=None, local_rank=None, num_workers=None):
        """
        Initialize a Trainer object.

        Args:
            dataset_kwargs (dict, optional): Keyword arguments for creating dataset object.
            load_data_kwargs (dict, optional): Keyword arguments for data loaders.
            model_kwargs (dict, optional): Keyword arguments for creating the model
            device (str, optional): Device to use for training. Defaults to None.
            work_dir (str, optional): Directory to save training outputs. Defaults to None.
        """
        if log_name == None:
            log_name = "training.log"
        if log_level == None:
            log_level = "INFO"
        self.work_dir = work_dir
        self.dataset_kwargs = dataset_kwargs
        self.load_data_kwargs = load_data_kwargs
        self.model_kwargs = model_kwargs
        self.device = device
        self.log_name = log_name
        self.log_level = getattr(logging, log_level)
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.local_rank = local_rank
        self.num_workers = num_workers
        self.force = force
        self.timing_name = timing_name
        self.mode_test = mode_test
        
        # === Deal with the configuration file === #
        if work_dir is not None:
            os.makedirs(os.path.dirname(self.work_dir), exist_ok=True)
            config_file = os.path.join(self.work_dir, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.warning(f"Config file {config_file} found -> loading configuration")
                #assert config['work_dir'] == self.work_dir, f"work_dir in the config file is different from the current work_dir: {config['work_dir']} != {self.work_dir}"
                # === Applying local changes to the configuration file === #
                # === this is useful if we would like to load traner object with different parameters === #
                # === and particularly useful when trainging and testing on different architectures === #
                if self.dataset_kwargs is not None:
                    config['dataset_kwargs'] = self.dataset_kwargs
                if self.load_data_kwargs is not None:
                    config['load_data_kwargs'] = self.load_data_kwargs
                if self.model_kwargs is not None:
                    config['model_kwargs'] = self.model_kwargs
                if self.mode_test is not None:
                    config['mode_test'] = self.mode_test
                if self.device is not None:
                    config['device'] = self.device
                if self.log_name is not None:
                    config['log_name'] = self.log_name
                if self.log_level is not None:
                    config['log_level'] = self.log_level
                if self.world_size is not None:
                    config['world_size'] = self.world_size
                if self.rank is not None:
                    config['rank'] = self.rank
                if self.gpus_per_node is not None:
                    config['gpus_per_node'] = self.gpus_per_node
                if self.local_rank is not None:
                    config['local_rank'] = self.local_rank
                if self.num_workers is not None:
                    config['num_workers'] = self.num_workers
                
                self.__dict__.update(**config) # update the attributes with the configuration file
            else:
                config = copy.deepcopy(self.__dict__) # save the attributes to the config file
                logger.info(f"Creating a new configuration file: {config_file}")
                os.makedirs(os.path.dirname(self.work_dir), exist_ok=True)
                try:
                    with open(config_file, 'w') as f:
                        f.write(json.dumps(config, indent=4))
                except Exception as e:
                    logger.error(f"Error saving configuration file: {e}")
        self.config = copy.deepcopy(self.__dict__) # save the attributes to the config of the trainer class 
        if self.work_dir is not None:
            self.f_handler = self.set_logger(f'{self.work_dir}/{self.log_name}')

        
        self.dataset_kwargs.pop('samples_file',None)  # guardrails against accidentally passing samples_file to DataFrameDataset     
        train_sample = self.dataset_kwargs.pop('train_sample')
        val_sample = self.dataset_kwargs.pop('val_sample')
        test_sample = self.dataset_kwargs.pop('test_sample')

        if not self.mode_test:
            self.train_dataset = datasets.DataFrameDataset(datalabel="train", 
                                                           samples_file=train_sample,
                                                           norm_folder=self.work_dir,
                                                       **self.dataset_kwargs)
            self.val_dataset = datasets.DataFrameDataset(datalabel="val",
                                                          samples_file=val_sample,
                                                           norm_folder=self.work_dir,
                                                       **self.dataset_kwargs)
                                                       
        self.test_dataset = datasets.DataFrameDataset(datalabel="test",
                                                          samples_file=test_sample,
                                                           norm_folder=self.work_dir,
                                                       **self.dataset_kwargs)
        
        self.comprehend_config()

    def create_empty_csv(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Host', 'Rank', 'Local Rank', 'Time'])
    def set_logger(self, log_dir=None):
        if log_dir is not None:
            # Remove all handlers associated with the root logger object.
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            
            # Set the log level to CRITICAL for the root logger
            logging.root.setLevel(self.log_level)
            
            f_handler = logging.FileHandler(log_dir)
            f_handler.setLevel(self.log_level)
            warnings_logger = logging.getLogger("py.warnings")
            job_id = ""
            if "SLURM_JOB_ID" in os.environ:
                job_id = f'job:{os.environ["SLURM_JOB_ID"]}'
            f_format = logging.Formatter('%(job_id)s | %(nodename)s | @rank: %(rank)s | @local: %(local_rank)s at %(asctime)s | %(levelname)s | %(name)s  | \t %(message)s')
            f_handler.setFormatter(f_format)

            # Add the custom filter to the handler
            custom_filter = CustomFilter(job_id, self.rank, self.local_rank, os.uname().nodename)
            f_handler.addFilter(custom_filter)

            logger.addHandler(f_handler)
            warnings_logger.addHandler(f_handler)
            logger.setLevel(self.log_level)
            logger.info(f" ")
            logger.info(f"===Logging to {log_dir} on level {self.log_level}, @ {self.rank=}, {self.local_rank=}, {self.device=} ===") 
            logger.info(f"host: {os.uname().nodename}")
            logger.info(f" ")
            # Define the extra loggers and add the same FileHandler to them
            datasets_logger = logging.getLogger(__name__) # TODO: Fix the names 
            self.apply_handler(datasets_logger, f_handler, self.log_level) 
            models_logger = logging.getLogger(models.__name__)
            self.apply_handler(models_logger, f_handler, self.log_level)
            read_pic_logger = logging.getLogger(datasets.__name__)
            self.apply_handler(read_pic_logger, f_handler, self.log_level)
        return f_handler

    def set_logge_old(self, log_dir=None):
        if log_dir is not None:
            f_handler = logging.FileHandler(log_dir)
            f_handler.setLevel(self.log_level)
            warnings_logger = logging.getLogger("py.warnings")
            job_id = ""
            if "SLURM_JOB_ID" in os.environ:
                job_id = f'job:{os.environ["SLURM_JOB_ID"]}'
            f_format = logging.Formatter('%(job_id)s | %(nodename)s | @rank: %(rank)s | @local: %(local_rank)s at %(asctime)s | %(levelname)s | %(name)s  | \t %(message)s')
            f_handler.setFormatter(f_format)

            # Add the custom filter to the handler
            custom_filter = CustomFilter(job_id, self.rank, self.local_rank, os.uname().nodename)
            f_handler.addFilter(custom_filter)

            logger.addHandler(f_handler)
            warnings_logger.addHandler(f_handler)
            logger.setLevel(self.log_level)
            logger.info(f" ")
            logger.info(f"===Logging to {log_dir} on level {self.log_level}, @ {self.rank=}, {self.local_rank=}, {self.device=} ===") 
            logger.info(f"host: {os.uname().nodename}")
            logger.info(f" ")
            # Define the extra loggers and add the same FileHandler to them
            datasets_logger = logging.getLogger(__name__) # TODO: Fix the names 
            self.apply_handler(datasets_logger, f_handler, self.log_level) 
            models_logger = logging.getLogger(models.__name__)
            self.apply_handler(models_logger, f_handler, self.log_level)
            read_pic_logger = logging.getLogger(datasets.__name__)
            self.apply_handler(read_pic_logger, f_handler, self.log_level)
        return f_handler
            
    def apply_handler(self, logger, f_handler, log_level):
        logger.addHandler(f_handler)
        logger.setLevel(log_level)
        return None

    def set_dataset(self,datalabel="test", samples_file=None):
        if datalabel == "train":
            self.train_dataset = datasets.DataFrameDataset(datalabel=datalabel, samples_file=samples_file,norm_folder=self.work_dir,
                                                       **self.dataset_kwargs)
            if self.dataset_kwargs.pop('scaler_features', None) is True: # removing these to avoid passing them to the validation and test datasets
                self.scaler_features = (self.train_dataset.features_mean,self.train_dataset.features_std)
            else: # if it is None or it is False we should not try to pass normalization of the train set to the val or test sets
                self.scaler_features = None
            if self.dataset_kwargs.pop('scaler_targets', None) is True:
                self.scaler_targets = (self.train_dataset.targets_mean,self.train_dataset.targets_std)
            else: # if it is None or it is False we should not try to pass normalization of the train set to the val or test sets
                self.scaler_targets = None
        elif datalabel == "val":
            self.val_dataset = datasets.DataFrameDataset(datalabel=datalabel, samples_file=samples_file,norm_folder=self.work_dir, 
                                            scaler_features=self.scaler_features, 
                                            scaler_targets=self.scaler_targets,
                                            **self.dataset_kwargs)
        elif datalabel == "test":
            self.test_dataset = datasets.DataFrameDataset(datalabel=datalabel,samples_file=samples_file,norm_folder=self.work_dir,
                                            scaler_features=self.scaler_features, 
                                            scaler_targets=self.scaler_targets,
                                             **self.dataset_kwargs)
        else:
            raise ValueError(f"Unknown datalabel: {datalabel}")
    
    def comprehend_config(self):
        """
        Comprehends the configuration settings for the model and its configs. The method deep copies
        the configuration settings and extracts the necessary parameters for the model. At the end it calls the method
        trainer::load_data to create/update the data loaders accordingly.

        Returns:
            None
        """
        logger.info("Comprehending the configuration settings") # it is called once during creation of the Trainer object and once during the fit method
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
            
            self.model = models.PyNet(**model_kwargs, rank=self.rank, local_rank=self.local_rank) 
            #logger.info(f"Successfully parsed the {model_name} class")
            logger.info(f"Creating object: {self.model} which contains {self.model.model} as the model")
        else: # if model is already instantiated
            logger.info(f"Model provided as an input {model_kwargs =}")
            self.model = model_kwargs
        logger.info(f"{self.device = } on {self.local_rank = }")
        
        logger.info(f"Code version git hash: {ut.get_git_revision_hash()}") # TODO: add this to the config file and raise error if it doesn't coincide wiht the current version
        if not self.mode_test:
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
        if self.world_size is not None and self.world_size > 1:
            sampler_type = 'distributed'
        else:
            sampler_type = 'serial'
        logger.info(f"Creating data loaders with {sampler_type = } because {self.world_size = }")
        if 'num_workers' in load_data_kwargs['train_loader_kwargs']:
            logger.info("Config file contains num_workers in train_loader_kwargs so ignoring the number of actual cpus")
            self.train_loader = datasets.ChannelDataLoader(self.train_dataset, sampler_type=sampler_type, world_size = self.world_size, 
                                                       rank = self.rank, gpus_per_node = self.gpus_per_node, local_rank = self.local_rank,
                                                       **load_data_kwargs['train_loader_kwargs'])
        else:
            self.train_loader = datasets.ChannelDataLoader(self.train_dataset, sampler_type=sampler_type, world_size = self.world_size, 
                                                       rank = self.rank, gpus_per_node = self.gpus_per_node, local_rank = self.local_rank,
                                                       num_workers=self.num_workers, **load_data_kwargs['train_loader_kwargs'])
        if 'num_workers' in load_data_kwargs['val_loader_kwargs']:
            logger.info("Config file contains num_workers in train_loader_kwargs so ignoring the number of actual cpus")
            self.val_loader = datasets.ChannelDataLoader(self.val_dataset,sampler_type=sampler_type, world_size = self.world_size, 
                                                       rank = self.rank, gpus_per_node = self.gpus_per_node, local_rank = self.local_rank,
                                                       **load_data_kwargs['val_loader_kwargs'])
        else:
            self.val_loader = datasets.ChannelDataLoader(self.val_dataset,sampler_type=sampler_type, world_size = self.world_size, 
                                                       rank = self.rank, gpus_per_node = self.gpus_per_node, local_rank = self.local_rank,
                                                        num_workers=self.num_workers, **load_data_kwargs['val_loader_kwargs'])

    def load_run(self, run):
        """
        Load the configuration file of a specific run, comprehend config the associated file associated 
        with the model and load model weights.
        """
        config_file = os.path.join(self.work_dir, run, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            if self.work_dir is not None:
                self.set_logger(f'{self.work_dir}/{run}/run.log')
            logger.info(f"==========Config file {config_file} found, logging to {self.work_dir}/{run}/run.log========")
            logger.warning(f"Loading configuration based on it and the associated model weights/training loss!")
            if config['dataset_kwargs'] != self.config['dataset_kwargs']:
                raise ValueError("The old and new config file have dataset_kwargs which are not consistent! You must create a new run")
            self.config  = config
            self.comprehend_config()
            model_file = f"{self.config['work_dir']}/{run}/model.pth"    # < ======= TODO: Add multiple models here
            loss_file = f"{self.config['work_dir']}/{run}/loss_dict.pkl"
            logger.info(f"Loading model weights from {model_file}")
            try:
                self.model.model.load_state_dict(torch.load(model_file, map_location=self.device))
            except RuntimeError:
                # torch.nn.parallel.DistributedDataParallel, which prefixes the keys with "module.".
                # Load the state dictionary
                state_dict = torch.load(model_file, map_location=self.device)
                # Remove 'module.' prefix from keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v
                # Load the modified state dictionary
                self.model.model.load_state_dict(new_state_dict)
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
            trial = new_config.pop('trial',None) # handling optuna trials # TODO: check that this is working in distributed mode

            for key in new_config['dataset_kwargs']:
                if key in new_config['dataset_kwargs'] and new_config['dataset_kwargs'][key] != self.config['dataset_kwargs'][key]:
                    logger.warning(f"{key} is inconsistent between self.config and new_config. Check that you are not mixing test/train/validation")
            logger.warning(f"============Updating the config with the new config============")
            self.config = new_config
            self.comprehend_config()
            if self.run != '': # if we are running a trial, we need to save config to subdirectory
                if self.rank == 0:  #self.local_rank == 0:
                    config_dir = os.path.join(self.work_dir, self.run)
                    config_file = os.path.join(config_dir, 'config.json')
                    if os.path.exists(f"{self.work_dir}/{self.run}/config.json"):
                        if self.force:
                            logger.warning(f"Config file {self.work_dir}/{self.run}/config.json already exists. Overwriting due to --force!")
                            shutil.rmtree(f"{self.work_dir}/{self.run}/")
                        else:
                            raise FileExistsError(f"Config file {self.work_dir}/{self.run}/config.json already exists. Overwriting not allowed!")
                    # Ensure the directory exists before writing the config file
                    os.makedirs(config_dir, exist_ok=True)
                    logger.info(f"Saving the new configuration to {config_file}")
                    try:
                        with open(config_file, 'w') as f:
                            f.write(json.dumps(self.config, indent=4))
                    except Exception as e:
                        logger.error(f"Error saving configuration file: {e}")
                else:
                    if not os.path.exists(f"{self.work_dir}/{self.run}/config.json"):
                        logger.warning(f"Config file {self.work_dir}/{self.run}/config.json not found, which means that the rank=0 did not yet save the configuration file")
                if self.work_dir is not None:
                        os.makedirs(os.path.dirname(f"{self.work_dir}/{self.run}/"), exist_ok=True)
                        self.set_logger(f'{self.work_dir}/{self.run}/run.log')

        # Getting % usage of virtual_memory ( 3rd field)
        logger.info(f'Prior to fit: RAM memory % used: {psutil.virtual_memory()[2]}, RAM Used (GB):, {psutil.virtual_memory()[3]/1000000000}, process RAM usage (GB): {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3}')
        best_loss = self.model.fit(self.train_loader, self.val_loader, trial=trial)   
        logger.info(f'After fit: RAM memory % used: {psutil.virtual_memory()[2]}, RAM Used (GB):, {psutil.virtual_memory()[3]/1000000000}, process RAM usage (GB): {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3}')
        if self.local_rank == 0:
            if self.work_dir is not None:
                logger.info(f"Saving the model weights and loss history to {self.work_dir}/{self.run}/")
                os.makedirs(os.path.dirname(f"{self.work_dir}/{self.run}/"), exist_ok=True)
                model_path = f"{self.work_dir}/{self.run}/model.pth"
                torch.save(self.model.model.state_dict(), model_path)
                loss_path = f"{self.work_dir}/{self.run}/loss_dict.pkl"
                with open(loss_path, 'wb') as f:
                    pickle.dump( {'train_loss': self.model.train_loss_,'val_loss': self.model.val_loss_,'time': self.model.total_time}, f)
       
        return best_loss

    def _get_device(self, device):
        """
        Get the device to use for training.
        """
        if device is None:
            if torch.cuda.is_available():
                logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                current_device = torch.cuda.current_device()
                logger.info(f"Currently using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
            else:
                logger.info("No CUDA GPU available. Using CPU.")
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Work Directory')
    parser.add_argument('--force', action=argparse.BooleanOptionalAction,
                        help='Force the training to start even if the run exists')
    parser.add_argument('--timing_name', type=str, default=False,
                        help='Name of the timing CSV file. If not provided no timing file will be created')
    parser.add_argument('--config', action='append', default=None, help="Update nested config keys. Use 'key.subkey=value' format")
    # example usage: python -m closure.src.trainers --config work_dir=work_dir --config run=run --config model_kwargs.model_name=ResNet --config model_kwargs.model_depth=18

    args = parser.parse_args()

    # Extract work_dir from config
    work_dir = None
    if args.config:
        for update in args.config:
            key, value = update.split("=", 1)  # Split at the first '='
            if key == "work_dir":
                work_dir = value
                break

    if work_dir is None:
        raise ValueError("work_dir must be specified in the --config argument")

    # Check if running in a distributed environment
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)

        dist.init_process_group("nccl", rank=rank, world_size=world_size) # initialize the process group

        if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
        print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}, \
                    gpus_per_node: {gpus_per_node}, num_workers: {num_workers}")
    else:
        # Single-node setup
        world_size = 1
        rank = 0
        local_rank = 0
        gpus_per_node = torch.cuda.device_count()
        num_workers = os.cpu_count()
        print(f"Running on a single node with {gpus_per_node} GPUs and {num_workers} CPU cores.")

    print(f"Creating Trainer object with work_dir={work_dir}")
    trainer = Trainer(work_dir=work_dir, world_size=world_size, rank=rank, gpus_per_node=gpus_per_node, 
                    local_rank=local_rank, num_workers=num_workers, force=args.force, timing_name=args.timing_name)

    if args.config is not None:
        config = copy.deepcopy(trainer.config)
        # Apply nested updates from --config
        for update in args.config:
            key, value = update.split("=", 1)  # Split at the first '='
            if key != "work_dir":  # Skip work_dir as it is already handled
                set_nested_config(config, key, value)
                print(f"Setting {key} to {value}")
        #print(config)
        trainer.fit(config=config)
    else:
        trainer.fit()

    dist.destroy_process_group()

if __name__ == '__main__':
    main()