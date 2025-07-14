
"""
trainers.py
Repo:       closure
Projects:   STRIDE, HELIOSKILL
Author:     George Miloshevich
Date:       2025
License:    MIT License
Description:
    This module defines a Trainer class for orchestrating machine learning protocols.

Usage in Python:
    from closure.src.trainers import Trainer
    trainer = Trainer(work_dir='path/to/work_dir', dataset_kwargs=dataset_kwargs, model_kwargs=model_kwargs)
    best_loss = trainer.fit()

Usage in Command Line:
    python -m src.trainers --config work_dir=path/to/work_dir --config run=run_name --config model_kwargs.model_name=ResNet


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

class Trainer:
    """
    A trainer class that manages configs/logging/datasets/dataloaders/models. 

    See: constructor of this class for the Args.

    Attributes:
        work_dir (str):                     Directory to save training outputs.
        dataset_kwargs (dict):              kwargs for creating dataset object.
        load_data_kwargs (dict):            kwargs for data loaders.
        model_kwargs (dict):                kwargs for creating the model.
        mode_test (bool):                   whether trainer is intended to not load data. 
            If True, it will not load train/val datasets accelerating the dataloading.
        device (str):                       Device to use for training.
        config (dict):                      A copy of the attributes of the Trainer object
            that will be saved to the config file.
        run (str):                          The run name. If provided trainer will save 
            the config file to a subdirectory with the run name.
        log_name (str):                     Name of the log file.
        log_level (str):                    Logging level.
        force (bool):                       Whether to force the training to start even if the run exists.
        timing_name (str):                  Name of the timing CSV file which times how long the training took.
            If not provided no timing file will be created.
        mode_test (bool):                   Whether trainer is intended to not load data.
        device (str):                       Device to use for training.
        config (dict):                      A copy of the attributes of the Trainer object 
            that will be saved to the config file.
        
        f_handler (FileHandler):            File handler for logging.
        num_workers (int):                  Number of workers for data loading.
        world_size (int):                   Number of processes in distributed training.
        rank (int):                         Rank of the current process in distributed training.
        gpus_per_node (int):                Number of GPUs per node in distributed training.
        local_rank (int):                   Local rank of the current process in distributed training.

        train_dataset (DataFrameDataset):   Training dataset.
        val_dataset (DataFrameDataset):     Validation dataset.
        test_dataset (DataFrameDataset):    Test dataset.
        train_loader (DataLoader):          Data loader for training dataset.
        val_loader (DataLoader):            Data loader for validation dataset.
        model (nn.Module):                  Model that fits the data.

    Methods:
        __init__(self, dataset_kwargs=None, load_data_kwargs=None, model_kwargs=None, device=None, work_dir=None):
            Initializes a Trainer object.
        comprehend_config(self):
            Comprehends the configuration settings for the model and its configs.
        load_data(self, load_data_kwargs):
            Creates the data loaders for training/validation and testing.
        load_run(self, run):
            Loads the configuration file of a specific run, comprehend config the associated 
            file associated with the model and load model weights.
        fit(self, config=None):
            Fits the model to the training data and returns the best loss.
    
    Notes:
        - The Trainer object manages a parent directory (the "trainer folder"), 
            specified by its work_dir attribute. You may train a model in that parent directory as well.
        - Each run (if provided in trainer.fit) is a subfolder inside this parent directory. 
            The subfolder is typically named after the run (e.g., work_dir/0, work_dir/1, etc.).
        - Each run subfolder contains its own config.json, model weights, logs, and results. 
            This allows you to keep results from different experiments (runs) organized and reproducible.
        - When you want to start a new run, you usually change a few keys in the config 
            (such as hyperparameters or target variables), and the Trainer will save 
            the new configuration and outputs in a new subfolder.
        - The Trainer can load a specific run using the load_run method, 
            which loads the config and model state from the corresponding subfolder.

    """
    def __init__(self, dataset_kwargs=None, load_data_kwargs=None, model_kwargs=None, mode_test=False,
                 device=None, work_dir=None, log_name=None, log_level=None, force=False, timing_name=None,
                 world_size=None, rank=None, gpus_per_node=None, local_rank=None, num_workers=None):
        """
        Initialize a Trainer object.
        Args:
            dataset_kwargs (dict, optional):    kwargs for creating dataset object.
            load_data_kwargs (dict, optional):  kwargs for data loaders.
            model_kwargs (dict, optional):      kwargs for creating the model.
            mode_test (bool, optional):         Whether trainer is intended to not load data. Defaults to False.
                This will not load train/val datasets accelerating the dataloading. 
            device (str, optional):             Device to use for training. Defaults to None.
            work_dir (str, optional):           Directory to save training outputs. 
                Defaults to None in which case it will not save anything.
            log_name (str, optional):           Name of the log file. Defaults to "training.log".
            log_level (str, optional):          Logging level. Defaults to "INFO".
            force (bool, optional):             Whether to force the training to start 
                even if the run exists. Defaults to False.
            timing_name (str, optional):        Name of the timing CSV file. 
                If not provided no timing file will be created.
            world_size (int, optional):         Number of processes in distributed training. 
                Defaults to None.
            rank (int, optional):               Rank of the current process in distributed training. 
                Defaults to None.
            gpus_per_node (int, optional):     Number of GPUs per node in distributed training. 
                Defaults to None.
            local_rank (int, optional):         Local rank of the current process in distributed training. 
                Defaults to None.
            num_workers (int, optional):        Number of workers for data loading. Defaults to None, 
                in which case it will use os.cpu_count().
        Raises:
            FileExistsError: If the config file already exists and overwriting is not allowed.
            Exception: If there is an error saving the configuration file.
            
        """
            # Set default values for optional parameters
        self.log_name = log_name or "training.log"
        self.log_level = getattr(logging, log_level or "INFO")
        self.num_workers = num_workers or os.cpu_count()
        
        # Store initialization parameters
        self.work_dir = work_dir
        self.dataset_kwargs = dataset_kwargs or {}
        self.load_data_kwargs = load_data_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.device = device
        self.mode_test = mode_test
        self.force = force
        self.timing_name = timing_name
        
        # Store distributed training parameters
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.local_rank = local_rank
        
        # Handle configuration file management
        self._handle_config_file()
        
        # Set up device and logging
        self._setup_device_and_logging()
        
        # Initialize datasets
        self._initialize_datasets()
        
        # Initialize model and data loaders
        self.comprehend_config()

    def fit(self, config=None):
        """
        Fits the model to the training data and returns the best loss, while saving the products 
        of the training process to disk. Note that if you want to use subfolder `runs` you need to provide
        the `run` key in the config dictionary, and call `trainer.fit(config=config)`
        Args:
            config (dict): A dictionary containing the configuration parameters for the training process. 
            If provided, the original config will be updated with the new config. If dataset_kwargs are
            not consistent between the original and new config warning will be raised 
            (see comprehend_config method).
        Returns:
            float: The best loss achieved during the training process.

        Raises:
            FileExistsError: If the config file already exists and overwriting is not allowed.
            Exception: If there is an error saving the configuration file.

        Notes:
            - If `config` is provided, the original configuration will be updated with the new 
                configuration before training.
            - If `self.run` is not empty, the new configuration will be saved to a subdirectory.
        """
        # Handle optuna trials
        trial = None

        if config is not None: 
            # Process new configuration
            new_config = copy.deepcopy(config) # to avoid changing the original dictionary
            trial = new_config.pop('trial',None) # handling optuna trials # TODO: check that this is working in distributed mode

            # Validate dataset consistency (all runs managed by the same Trainer should have the same dataset_kwargs)
            self._validate_dataset_consistency(new_config)

            logger.warning(f"============Updating the config with the new config============")
            self.config = new_config
            self.comprehend_config() # update the model and data loaders consistently with the new config

            # Handle run-specific configurations (trainer folder contains subfolders for each run containg new config)
            if self.run != '':
                self._handle_run_config_save()

        self._log_memory_usage("Prior to fit")

        best_loss = self.model.fit(self.train_loader, self.val_loader, trial=trial)   

        self._log_memory_usage("After fit")
        # Save model and results (only on rank 0 in distributed training)
        self._save_training_results()
       
        return best_loss

    def comprehend_config(self):
        """
        Comprehends the configuration settings for the model and its configs. The method deep copies
        the configuration settings and extracts the necessary parameters for the model. 
        It is also responsible for creating the model object based on the provided
        model_kwargs. If the model is already instantiated, it will use the provided model object.
        If the model_kwargs is a dictionary, it will create a new model instance using the PyNet class.
        At the end it calls the method trainer::load_data to create/update the data loaders accordingly.

        Returns:
            None
        """
        logger.info("Comprehending the configuration settings") 
        config = copy.deepcopy(self.config)
        #dataset_kwargs = config['dataset_kwargs']
        load_data_kwargs = config['load_data_kwargs']
        model_kwargs = config['model_kwargs']
        #device = config['device']
        if 'run' in config:  # this is to handle optuna trials or other runs
            self.run = config['run']
        else:
            self.run = ''
        if isinstance(model_kwargs, dict): # if model is not instantiated we create it
            logger.warning(f"Creating new model. Note this will replace any previous model")
            self.model = models.PyNet(**model_kwargs, rank=self.rank, local_rank=self.local_rank) 
        else: # if model is already instantiated
            logger.info(f"Model provided as an input {model_kwargs =}")
            self.model = model_kwargs

        logger.info(f"{self.device = } on {self.local_rank = }")
        logger.info(f"Code version git hash: {ut.get_git_revision_hash()}") # TODO: add this to the config file and raise error if it doesn't coincide wiht the current version
        self.load_data(load_data_kwargs)
    
    def load_data(self, load_data_kwargs):
        """
        Creates the data loaders for training/validation and testing. Note that datasets have already been 
            created in the __init__ method. Depending on world_size, it will create either a distributed or 
            serial sampler for the data loaders.

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
        for phase in ['train', 'val', 'test']:
            if phase == 'test' or not self.mode_test:
                dataset = getattr(self, f"{phase}_dataset")
                if phase == 'test':
                    loader_kwargs = load_data_kwargs[f"val_loader_kwargs"]
                else:
                    loader_kwargs = load_data_kwargs[f"{phase}_loader_kwargs"]
                if 'num_workers' in loader_kwargs:
                    logger.info(f"Config file contains num_workers in {phase}_loader_kwargs so ignoring the number of actual cpus")
                    loader = datasets.ChannelDataLoader(dataset, sampler_type=sampler_type, 
                            world_size=self.world_size, rank=self.rank, gpus_per_node=self.gpus_per_node, 
                            local_rank=self.local_rank, **loader_kwargs)
                else:
                    loader = datasets.ChannelDataLoader(dataset, sampler_type=sampler_type, 
                            world_size=self.world_size, rank=self.rank, gpus_per_node=self.gpus_per_node, 
                            local_rank=self.local_rank, num_workers=self.num_workers, **loader_kwargs)
                setattr(self, f"{phase}_loader", loader)

    def load_run(self, run):
        """
        Load the configuration file of a specific run, comprehend config the associated file associated 
        with the model and load model weights. 

        Args:
            run (str): The name of the run to load. This should be a subdirectory in the work_dir.
        Raises:
            FileNotFoundError:  If the config file for the specified run does not exist.
            ValueError:         If the dataset_kwargs in the old and new config file are not consistent.
                This is done to ensure that all runs within trainer parent folder share the same dataset_kwargs.
                The only exception is if the transform is different, in which case it will not raise an error.
        """
        config_file = os.path.join(self.work_dir, run, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            if self.work_dir is not None:
                self.set_logger(f'{self.work_dir}/{run}/run.log', console_stream=False)
            logger.info(f"==========Config file {config_file} found, logging to {self.work_dir}/{run}/run.log========")
            logger.warning(f"Loading configuration based on it and the associated model weights/training loss!")
            
            if 'transform' in config['dataset_kwargs'] and 'transform' in self.config['dataset_kwargs']:
                for key in config['dataset_kwargs']:
                    if key != 'transform' and config['dataset_kwargs'][key] != self.config['dataset_kwargs'][key]:
                        raise ValueError(f"The old and new config file have dataset_kwargs which are not consistent in {key = }! You must create a new run")
            else:
                if config['dataset_kwargs'] != self.config['dataset_kwargs']:
                    raise ValueError("The old and new config file have dataset_kwargs which are not consistent! You must create a new run")
            self.config  = config
            self.comprehend_config()
            self.model.load(f"{self.config['work_dir']}/{run}")
        else:
            raise FileNotFoundError(f"Config file {config_file} not found.")
        
    def _initialize_datasets(self):
        """Initialize training, validation, and test datasets.
        This method creates the datasets based on the provided dataset_kwargs.
        It extracts the sample file paths for training, validation, and testing datasets,
        and creates DataFrameDataset objects for each dataset.
        If the mode_test is True, it will only create the test dataset (useful for inference mode testing).
        If mode_test is False, it will create train and validation datasets as well."""
        # Remove 'samples_file' from dataset_kwargs to prevent conflicts
        self.dataset_kwargs.pop('samples_file', None)
        
        # Extract sample file paths
        train_sample = self.dataset_kwargs.pop('train_sample')
        val_sample = self.dataset_kwargs.pop('val_sample')
        test_sample = self.dataset_kwargs.pop('test_sample')
        
        
        # Create train and validation datasets only if not in test mode
        if not self.mode_test:
            self.train_dataset = datasets.DataFrameDataset(
                datalabel="train",
                samples_file=train_sample,
                norm_folder=self.work_dir,
                **self.dataset_kwargs
            )
            
            self.val_dataset = datasets.DataFrameDataset(
                datalabel="val",
                samples_file=val_sample,
                norm_folder=self.work_dir,
                **self.dataset_kwargs
            )
        
        # Always create test dataset
        self.test_dataset = datasets.DataFrameDataset(
            datalabel="test",
            samples_file=test_sample,
            norm_folder=self.work_dir,
            **self.dataset_kwargs
        )

    def _handle_config_file(self):
        """ Handle the configuration file for the Trainer.
        This method checks if a configuration file already exists in the work directory.
        If it does, it loads the existing configuration and updates it with any new values provided during
        initialization of the Trainer. If the configuration file does not exist, it creates a new one with the current
        configuration settings. The final configuration is stored in the `self.config` attribute of the Trainer.
        """
        if self.work_dir is None:
            self.config = copy.deepcopy(self.__dict__)
            return
        
        # Ensure work directory exists
        os.makedirs(os.path.dirname(self.work_dir), exist_ok=True)
        config_file = os.path.join(self.work_dir, 'config.json')
        
        if os.path.exists(config_file):
            self._load_existing_config(config_file)
        else:
            self._create_new_config(config_file)
        
        # Store final configuration
        self.config = copy.deepcopy(self.__dict__)

    def _load_existing_config(self, config_file):
        """Load existing configuration file and update with new values
        that were provided to the constructer of the Trainer class."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            logger.warning(f"Config file {config_file} found -> loading configuration")
            
            # Update config with any new values provided during initialization
            updatable_keys = [
                'dataset_kwargs', 'load_data_kwargs', 'model_kwargs', 'mode_test', 'device',
                'log_name', 'log_level', 'world_size', 'rank', 'gpus_per_node', 'local_rank', 'num_workers'
            ]
            
            for key in updatable_keys:
                value = getattr(self, key, None)
                if value is not None:
                    config[key] = value
            
            # Update trainer object attributes with loaded config
            self.__dict__.update(**config)
            
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file}: {e}")
            raise

    def _create_new_config(self, config_file):
        """Save configuration file to disk."""
        config = copy.deepcopy(self.__dict__)
        logger.info(f"Creating a new configuration file: {config_file}")
        
        try:
            with open(config_file, 'w') as f:
                f.write(json.dumps(config, indent=4))
        except Exception as e:
            logger.error(f"Error saving configuration file: {e}")
            raise

    def _setup_device_and_logging(self):
        """Set up device selection and logging (cpu or cuda)."""
        # Force CPU if CUDA is not available
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        
        # Set up logging if work directory is provided
        if self.work_dir is not None:
            self.f_handler = self.set_logger(f'{self.work_dir}/{self.log_name}')
    
    def _validate_dataset_consistency(self, new_config):
        """Validate that dataset_kwargs are consistent between configs to avoid re-using wrong X.pkl normalization.
        all runs managed by the same Trainer should have the same dataset_kwargs.
        """
        for key in new_config['dataset_kwargs']:
            if (key in new_config['dataset_kwargs'] and 
                new_config['dataset_kwargs'][key] != self.config['dataset_kwargs'][key]):
                logger.warning(
                    f"{key} is inconsistent between self.config and new_config. "
                    "Check that you are not mixing test/train/validation"
                )
    
    def _handle_run_config_save(self):
        """Handle saving configuration for specific runs.
        This method handles the saving of the configuration file for a specific run.
        It checks if the run directory already exists and whether to overwrite it based on the `force
        flag. If the run directory does not exist, it creates it and saves the configuration file.
        If the run directory exists and `force` is set to True, it will overwrite the
            existing directory and save the new configuration file. If `force` is not set, it raises a
            FileExistsError to prevent overwriting existing configurations.
        It also sets up logging for the run, ensuring that logs are saved to a file in the run directory.
            """
        if self.rank is None or self.rank == 0:  # Only rank 0 saves the config file
            config_dir = os.path.join(self.work_dir, self.run)
            config_file = os.path.join(config_dir, 'config.json')
            
            # Check if config already exists
            if os.path.exists(config_file):
                if self.force:
                    logger.warning(
                        f"Config file {config_file} already exists. "
                        "Overwriting due to --force!"
                    )
                    shutil.rmtree(os.path.join(self.work_dir, self.run))
                else:
                    raise FileExistsError(
                        f"Config file {config_file} already exists. "
                        "Overwriting not allowed!"
                    )
            
            # Create directory and save config
            os.makedirs(config_dir, exist_ok=True)
            logger.info(f"Saving the new configuration to {config_file}")
            
            try:
                with open(config_file, 'w') as f:
                    f.write(json.dumps(self.config, indent=4))
            except Exception as e:
                logger.error(f"Error saving configuration file: {e}")
        else:
            # Non-rank-0 processes check if config exists
            config_file = os.path.join(self.work_dir, self.run, 'config.json')
            if not os.path.exists(config_file):
                logger.warning(
                    f"Config file {config_file} not found, which means that "
                    "rank=0 did not yet save the configuration file"
                )
        
        # Set up logging for the run
        if self.work_dir is not None:
            os.makedirs(os.path.dirname(f"{self.work_dir}/{self.run}/"), exist_ok=True)
            self.set_logger(f'{self.work_dir}/{self.run}/run.log')

    def _log_memory_usage(self, phase):
        """Log current memory usage."""
        memory_percent = psutil.virtual_memory()[2]
        memory_gb = psutil.virtual_memory()[3] / 1000000000
        process_memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        
        logger.info(
            f'{phase}: RAM memory % used: {memory_percent}, '
            f'RAM Used (GB): {memory_gb}, '
            f'process RAM usage (GB): {process_memory_gb}'
        )

    def _save_training_results(self):
        """Save model weights and training history.
        This method saves the model weights and training history to a specified directory.
        It checks if the local rank is None or 0 to ensure that only one process saves the results.
        If the work directory is specified, it creates a subdirectory for the run and
        saves the model weights and loss history in that directory."""
        if self.local_rank is None or self.local_rank == 0:
            if self.work_dir is not None:
                save_dir = os.path.join(self.work_dir, self.run)
                logger.info(f"Saving the model weights and loss history to {save_dir}/")
                
                # Create directory
                os.makedirs(save_dir, exist_ok=True)
                
                # Save model weights
                model_path = os.path.join(save_dir, 'model.pth')
                torch.save(self.model.model.state_dict(), model_path)
                
                # Save loss history
                loss_path = os.path.join(save_dir, 'loss_dict.pkl')
                loss_data = {
                    'train_loss': self.model.train_loss_,
                    'val_loss': self.model.val_loss_,
                    'time': self.model.total_time
                }
                with open(loss_path, 'wb') as f:
                    pickle.dump(loss_data, f)

    def _get_device(self, device):
        """
        Get the device to use for training.
        """
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
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

    def set_logger(self, log_dir=None, console_stream=True):
        """
        Configures logging for the trainer, including file and stream handlers, custom formatting, 
        and logger levels. If a log directory is provided, this method sets up a FileHandler to log messages 
        to the specified file, applies a custom formatter and filter to include job, node, and rank information,
        and attaches the handler to the main logger, warnings logger, and other relevant loggers. It also 
        sets up a StreamHandler to output logs to the console (useful for notebooks or interactive sessions).
        Args:
            log_dir (str, optional): Path to the log file. If None, only console logging is configured.
            console_stream (bool, optional): Whether to add a StreamHandler for console output. Defaults to True.
        Returns:
            logging.FileHandler: The file handler used for logging to the specified file, or None if 
            log_dir is not provided.
        """

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
            f_format = ut.SafeFormatter('%(job_id)s | %(nodename)s | @rank: %(rank)s | @local: %(local_rank)s at %(asctime)s | %(levelname)s | %(name)s  | \t %(message)s')
            f_handler.setFormatter(f_format)

            # Add the custom filter to the handler
            custom_filter = ut.CustomFilter(job_id, self.rank, self.local_rank, os.uname().nodename)
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
        # Add a StreamHandler to also output logs to the console (and thus to the notebook)
        stream_format = ut.SafeFormatter('%(asctime)s | %(levelname)s | %(name)s | \t %(message)s')
        if console_stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.log_level)
            stream_handler.setFormatter(stream_format)
        
        

        # Optionally, set propagate to False for your main loggers
        logger.propagate = False
        datasets_logger.propagate = False
        models_logger.propagate = False
        read_pic_logger.propagate = False
        #logger.addHandler(stream_handler)
        #warnings_logger.addHandler(stream_handler)
        #datasets_logger.addHandler(stream_handler)
        #models_logger.addHandler(stream_handler)
        #read_pic_logger.addHandler(stream_handler)
         
        ## Attach handlers to the root logger
        #logging.root.addHandler(f_handler)
        #logging.root.addHandler(stream_handler)

        # Attach to all known loggers
        for logger_name in logging.Logger.manager.loggerDict:
            log = logging.getLogger(logger_name)
            self.add_handler_once(log, f_handler)
            self.add_handler_once(log, stream_handler)
            log.setLevel(self.log_level)
        
       

        return f_handler
    
    def add_handler_once(self,logger, handler):
        if handler not in logger.handlers:
            logger.addHandler(handler)
            
    def apply_handler(self, logger, f_handler, log_level):
        logger.addHandler(f_handler)
        logger.setLevel(log_level)
        return None

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
        gpus_per_node = min(torch.cuda.device_count(),1) if torch.cuda.is_available() else 0
        num_workers = os.cpu_count() if os.cpu_count() < 32 else 32
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
                ut.set_nested_config(config, key, value)
                print(f"Setting {key} to {value}")
        #print(config)
        trainer.fit(config=config)
    else:
        trainer.fit()

    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"Error destroying process group, possibly because it didn't exist?")
        print(e)

if __name__ == '__main__':
    main()