
"""
datasets.py
This module provides custom dataset and data loading utilities for closure repo workflows,
particularly for distributed and channel-based data loading scenarios. It is designed to work
with PyTorch and supports advanced features such as distributed sampling, subsampling, patch-based
cropping, feature/target channel selection, and normalization.
Classes:
    - DistributedSampler: A sampler that restricts data loading to a subset of the dataset based on provided indices,
      supporting distributed training with PyTorch's DistributedDataParallel.
    - SubSampler: A custom sampler that allows for optional shuffling and subsampling of dataset indices.
    - ChannelDataLoader: An extension of PyTorch's DataLoader that supports channel-based data loading, subsampling,
      distributed sampling, and patch extraction from images.
    - DataFrameDataset: A dataset class for loading data from a DataFrame, supporting feature/target normalization,
      filtering, prescaling, and transformation.
Key Features:
    - Distributed and serial sampling for efficient data loading in multi-GPU or multi-node environments.
    - Flexible subsampling and shuffling of dataset indices for training and evaluation.
    - Channel selection for both features and targets, allowing for fine-grained control over input/output data.
    - Patch-based cropping for image data, enabling random spatial sampling during training.
    - Support for feature and target normalization, including pre-scaling and saving/loading of normalization parameters.
    - Logging for key operations and warnings to aid in debugging and reproducibility.
Intended Usage:
    - Designed for use in machine learning pipelines where data is stored as images or arrays, and metadata is managed
      via CSV files or DataFrames.
    - Suitable for both single-node and distributed training scenarios.
    - Can be extended or customized for specific project requirements.
Dependencies:
    - numpy, pandas, torch, scipy, joblib
Author: George Miloshevich
License: MIT License
Repo:       closure
Projects:   STRIDE, HELIOSKILL
Author:     George Miloshevich
Date:       2025
License:    MIT License
Description:
    
"""

import numpy
import torch
import os
from typing import Any, Tuple, TypeVar

import pandas as pd
import numpy as np
import joblib
import scipy.ndimage as nd
from torchvision.transforms import v2


from  . import read_pic as rp

import logging
logger = logging.getLogger(__name__)


import copy

T_co = TypeVar('T_co', covariant=True)

class DataFrameDataset(torch.utils.data.Dataset):
    
    """
    A custom PyTorch dataset class for loading data from a DataFrame.

    DataFrameDataset is a custom PyTorch Dataset for loading and preprocessing data from a DataFrame, 
    typically used for supervised learning tasks involving image or array data stored as files. 
    It supports flexible feature/target selection, normalization, filtering, and transformation pipelines.
    This class is designed to:
    - Load sample metadata (which provides datasplit information) from a CSV file into a DataFrame.
    - Read features and targets from disk using filenames listed in the DataFrame.
    - Optionally filter, pre-scale, and normalize features and targets using user-specified or precomputed statistics.
    - Support both flattened and channel-first (NCHW) data formats to accommodate different model requirements, e.g.
        we expect local model to operate with flattened pixel-wise data, while convolutional models to treat fields
        like images, i.e. with channels first (NCHW).
    - Apply torchvision-style transforms to features and targets, with support for deterministic application.

    See constructor of this class for the Args.

    Attributes:
        targets_dtype (torch.dtype):             The data type of the targets when __getitem__ is called by data loader
        targets_dtype_numpy (numpy.dtype):       The original pre-processed data type of the targets in numpy format
        features_dtype (torch.dtype):            The data type of the features when __getitem__ is called by data loader
        features_dtype_numpy (numpy.dtype):      The original pre-processed data type of the features in numpy format.
        scaler_features (tuple or None):        The scaler (normalization) applied after pre-scaler to features.
        scaler_targets (tuple or None):         The scaler (normalization) applied after pre-scaler to targets.
        prescaler_features (list or None):      The pre-scaler functions (such as log) to apply to the features.
        prescaler_targets (list or None):       The pre-scaler functions (such as log) to apply to the targets.
        samples_file (str or None):             The CSV file containing the sample filenames which provides 
            the sample filenames that are extracted from data_folder. samples_file is used to create dataframe ->
        dataframe (pd.DataFrame):               The DataFrame containing the sample filenames that are extracted from 
            data_folder to create the full dataset by concatenating each file labelled in consecutive raws of dataframe
        image_file_name_column (str):           The column name in the DataFrame that contains the image filenames.
        data_folder (str):                      The folder where the input data is stored. Which data is used
            is controlled by samples_file, which is a CSV file containing the filenames of the data.
        norm_folder (str):                      The folder to save the normalization parameters (mean and std) 
            for features and targets. This is used if scaler_features or scaler_targets are provided.
        read_features_targets_kwargs (dict):    Additional keyword arguments to pass to the `read_features_targets` 
            function. Examples:
                {'fields_to_read' :    {"B": True,"B_ext": False,"divB": False,"E": True,"E_ext": False,"rho": True,
                    "J": True, "P": True,"PI": True,"Heat_flux": False,"N": False,"Qrem": False}, # which fields to read
                'request_features' :  ['rho_e', 'Bx', 'By', 'Bz', 'Vx_e', 'Vy_e', 'Vz_e', 'Ex', 'Ey', 'Ez'], # input
                'request_targets' :   ["Pxx_e", "Pyy_e","Pzz_e","Pxy_e","Pxz_e","Pyz_e"],  # what we want to predict
                'choose_species' :    ['e',None],   # which species to load/omit.
                'choose_x' : [0,256], 'choose_y' : [0,256], 'verbose' : False
        logger:                                 The logger object for logging messages.
        
        flatten (bool):                         Whether to flatten the features and targets. This is needed
            when treating each pixel as individual sample and applying say MLP model to it. Default is True.
        features (np.ndarray):                  The features of the dataset.
        targets (np.ndarray):                   The targets of the dataset.
        features_shape (tuple):                 The shape of the features.
        targets_shape (tuple):                  The shape of the targets.
        samples (int):                          The number of samples in the dataset.
        request_features (list or None):        The requested features to load.
        request_targets (list or None):         The requested targets to load.
        filter_features (dictionary or None):   The filter to apply to the features.
        filter_targets (dictionary or None):    The filter to apply to the targets.
        transform (dictionary or None):         The transform to apply to the features, only the train set.

    Example:
        Example usage:
        dataset = DataFrameDataset(
            data_folder='/path/to/data',
            norm_folder='/path/to/norm',
            samples_file='/path/to/samples.csv',
            features_dtype='float32',
            targets_dtype='float32',
            scaler_features=None,
            scaler_targets=None,
            transform={'RandomCrop': {'size': (16, 16)}, 'apply': ['train']},
            datalabel='train'
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    Methods:
        load_original(): Loads the DataFrame from a CSV file and prepares the features and targets for further processing.
        scale_data(): Scales the features and targets of the dataset using pre-defined scalers or calculates 
            and saves new scalers if necessary.
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the features and targets for a given index.

    """
    def __init__(self, data_folder: str, norm_folder: str, samples_file: str = None,
                     features_dtype: str = 'float32', feature_dtype: str = None,  # Accept both names
                     targets_dtype: str = 'float32', target_dtype: str = None,  # Accept both names
                     features_dtype_numpy: str = 'float64', feature_dtype_numpy: str = None, # Accept both names
                     targets_dtype_numpy: str = 'float64', target_dtype_numpy: str = None,  # Accept both names
                     prescaler_features: list = None, prescaler_targets: list = None,
                     scaler_features: bool = None, scaler_targets: bool = None,
                     datalabel: str = 'train', flatten: bool = True,
                     image_file_name_column: str = 'filenames',
                     read_features_targets_kwargs: dict = None,
                     filter_features: dict = None, filter_targets: dict = None,
                     transform: dict = None):
            """
            Args:
                data_folder (str): The folder where the images are stored.
                norm_folder (str): The folder to save the normalization parameters.
                samples_file: CSV file with sample filenames and metadata
                features_dtype (str, optional): The data type of the features when __getitem__ is called. 
                    Defaults to 'float32'.
                feature_dtype (str, optional): Alternative name for features_dtype.
                targets_dtype (str, optional): The data type of the targets when __getitem__ is called. 
                    Defaults to 'float32'.
                datalabel: Dataset split label ('train', 'val', 'test')
                flatten: If True, flatten spatial dimensions for pixel-wise processing
                image_file_name_column: Column name in CSV containing filenames
                features_dtype_numpy (str, optional): The data type of the features. Defaults to 'float32'.
                feature_dtype_numpy (str, optional): Alternative name for features_dtype_numpy.
                target_dtype_numpy (str, optional): The data type of the targets. Defaults to 'float32'.
                samples_file (str, optional): The file containing the sample filenames. Defaults to None.
                prescaler_features (str, optional): The pre-scaler function to apply to the features. Defaults to None.
                prescaler_targets (str, optional): The pre-scaler function to apply to the targets. Defaults to None.
                scaler_features (tuple or None, optional): The scaler for features. If a tuple is provided, 
                    it should contain the mean and standard deviation of the features. Defaults to None.
                scaler_targets (tuple or None, optional): The scaler for targets. If a tuple is provided, 
                    it should contain the mean and standard deviation of the targets. Defaults to None.
                image_file_name_column (str, optional): The column name in the DataFrame that contains the image filenames. 
                    Defaults to 'filenames'.
                read_features_targets_kwargs (dict, optional): Additional keyword arguments to pass to the 
                    `read_features_targets` function. Defaults to None.
                filter_features (str, optional): The filter to apply to the features. Defaults to None.
                filter_targets (str, optional): The filter to apply to the targets. Defaults to None.
                transform: Data augmentation transforms (applied only to specified splits)
            """
            # Accept both features_dtype and feature_dtype, with feature_dtype taking precedence if provided
            if feature_dtype is not None:
                features_dtype = feature_dtype
            if feature_dtype_numpy is not None:
                features_dtype_numpy = feature_dtype_numpy
            if target_dtype is not None:
                targets_dtype = target_dtype
            if target_dtype_numpy is not None:
                targets_dtype_numpy = target_dtype_numpy

            # Store basic configuration
            self.data_folder = data_folder
            self.norm_folder = norm_folder
            self.samples_file = samples_file
            self.datalabel = datalabel
            self.flatten = flatten
            self.image_file_name_column = image_file_name_column
            self.logger = logger
            # Extract feature and target channel names
            self.read_features_targets_kwargs = read_features_targets_kwargs or {}
            self.request_features = self.read_features_targets_kwargs.get('request_features', None)
            self.request_targets = self.read_features_targets_kwargs.get('request_targets', None)

            logger.info(f" This is {self.datalabel} set")
            
            # Configure data types
            self._setup_data_types(features_dtype, targets_dtype, features_dtype_numpy, targets_dtype_numpy)
            
            # Configure preprocessing options
            self._setup_preprocessing(prescaler_features, prescaler_targets, scaler_features, scaler_targets)
            
            # Configure filtering
            self._setup_filtering(filter_features, filter_targets)
            
            # Configure transforms
            self._setup_transforms(transform)
            
            # Load and process data
            self.load_original()
            self.scale_data()
    
    def load_original(self):
        """Load data from files specified in the CSV samples file, which splits 
        the data into train, validation, and test sets. Request to load features and targets
        specified in read_features_targets_kwargs and reshape according to flatten condition."""
        logger.info(f"Loading data split from: {self.samples_file}")
        
        # Load sample metadata
        self.dataframe = pd.read_csv(self.samples_file)
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.filenames = self.dataframe[self.image_file_name_column].tolist()
        
        
        # Load features and targets from files
        self.features, self.targets = rp.read_features_targets(
            self.data_folder, self.filenames,
            features_dtype=self.features_dtype_numpy,
            targets_dtype=self.targets_dtype_numpy,
            **self.read_features_targets_kwargs
        )
        
        # Apply filtering if configured
        if self.filter_features is not None:
            self.features = self.filter_features(self.features, **self.filter_features_kwargs)
        
        if self.filter_targets is not None:
            self.targets = self.filter_targets(self.targets, **self.filter_targets_kwargs)
        
        # Store original shapes
        self.features_shape = self.features.shape
        self.targets_shape = self.targets.shape
        
        # Reshape data based on processing mode
        if self.flatten:
            # Flatten for pixel-wise processing (MLP models)
            self.features = self.features.reshape(-1, self.features.shape[-1])
            self.targets = self.targets.reshape(-1, self.targets.shape[-1])
        else:
            # Convert to channel-first format for CNN models (NCHW)
            self.features = self.features.transpose(0, 3, 1, 2)
            self.targets = self.targets.transpose(0, 3, 1, 2)
        
        logger.info(f"Data shape - Features: {self.features.shape}, Targets: {self.targets.shape}")
        self.samples = self.features.shape[0]

    def scale_data(self): 
        """
        Scales the features and targets of the dataset using pre-defined scalers or 
        calculates and saves new scalers if necessary.

        This method performs the following steps:
        1. If pre-scalers for features are provided, applies the pre-scalers to each channel of the features.
        2. If scalers for features are provided as a tuple, this comes in the form of passing the mean and standard deviation 
        of the features which will be used for the processing. This is a handy feature if these quantities have been precomputed
        on a train set and now have to be applied to validation set. If this tuple is not provided but scalers are not set to None
            then the script checks if if the file already exists, loads the mean and standard deviation from the file. If it
            does not exist then it calculates the mean and standard deviation of the features and saves them to a file.
        3. Normalizes the features by subtracting the mean and dividing by the standard deviation for each channel.
        
        Repeat 1-3 for the targets.
        """
        # Process features
        self._apply_prescaling(self.features, self.prescaler_features, "features")
        self._apply_normalization(self.features, "features")
        
        # Process targets
        self._apply_prescaling(self.targets, self.prescaler_targets, "targets")
        self._apply_normalization(self.targets, "targets")
        
        # Convert to PyTorch tensors
        self.features = torch.tensor(self.features, dtype=self.features_dtype)
        self.targets = torch.tensor(self.targets, dtype=self.targets_dtype)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Tells data loaders how to load the data from the dataset. If transform is not None,
        it applies the transform to both features and targets. This is useful for data augmentation
        Args:
            idx (int): The index of the sample to load.
        Returns:
            Tuple[Any, Any]: A tuple containing the features and targets for the given index.
        """        
        features, targets = self.features[idx], self.targets[idx]
        if self.transform is not None:
            state = torch.get_rng_state()
            features = self.transform(features)
            torch.set_rng_state(state) # to make sure that the same transform is applied to both features and targets in case of randomness
            targets = self.transform(targets)
        return features, targets
    
    def _apply_prescaling(self, data, prescaler_functions, data_type):
        """Apply pre-scaling functions (e.g., log transform) to each channel."""
        if prescaler_functions is None:
            return
        
        for channel in range(data.shape[1]):
            if prescaler_functions[channel] is not None:
                data[:, channel, ...] = prescaler_functions[channel](data[:, channel, ...])
                logger.info(f"Applied {prescaler_functions[channel].__name__} to {data_type} channel {channel}")
    
    def _apply_normalization(self, data, data_type):
        """Apply mean-std normalization to each channel."""
        scaler_enabled = getattr(self, f'scaler_{data_type}')
        
        if not scaler_enabled:
            return
        
        # Load or compute normalization parameters
        mean, std = self._get_normalization_params(data, data_type)
        
        # Apply normalization
        logger.info(f"Applying normalization to {data_type}")
        for channel in range(data.shape[1]):
            try:
                data[:, channel, ...] -= mean[channel]
                data[:, channel, ...] /= std[channel]
            except Exception as e:
                logger.error(f"Normalization failed for {data_type} channel {channel}")
                logger.error(f"Data shape: {data.shape}, Mean shape: {mean.shape}")
                raise e
        
        # Store normalization parameters
        setattr(self, f'{data_type}_mean', mean)
        setattr(self, f'{data_type}_std', std)
    
    def _get_normalization_params(self, data, data_type):
        """Load existing normalization parameters or compute new ones."""
        dtype_numpy = getattr(self, f'{data_type}_dtype_numpy')
        filename = f'{self.norm_folder}/{"X" if data_type == "features" else "y"}.pkl'
        
        if os.path.exists(filename):
            mean, std = joblib.load(filename)
            logger.info(f"Loaded normalization parameters for {data_type} from {filename}")
            return mean, std
        
        if self.datalabel != 'train':
            raise ValueError(
                f"Normalization parameters for {data_type} not found at {filename}. "
                "Parameters must be computed on training data first."
            )
        
        # Compute normalization parameters
        if len(data.shape) > 2:
            # For image data, compute statistics across batch and spatial dimensions
            mean = np.mean(data, axis=(0, 2, 3)).astype(dtype_numpy)
            std = np.std(data, axis=(0, 2, 3)).astype(dtype_numpy)
        else:
            # For flattened data, compute statistics across batch dimension
            mean = np.mean(data, axis=0).astype(dtype_numpy)
            std = np.std(data, axis=0).astype(dtype_numpy)
        
        # Save parameters
        os.makedirs(self.norm_folder, exist_ok=True)
        joblib.dump((mean, std), filename)
        logger.info(f"Computed and saved normalization parameters for {data_type} to {filename}")
        
        return mean, std

    def _setup_data_types(self, features_dtype, targets_dtype, features_dtype_numpy, targets_dtype_numpy):
        """Configure PyTorch and NumPy data types for features and targets."""
        self.features_dtype = getattr(torch, features_dtype)
        self.targets_dtype = getattr(torch, targets_dtype)
        self.features_dtype_numpy = getattr(numpy, features_dtype_numpy)
        self.targets_dtype_numpy = getattr(numpy, targets_dtype_numpy)
    
    def _setup_preprocessing(self, prescaler_features, prescaler_targets, scaler_features, scaler_targets):
        """Configure pre-scaling and normalization options."""
        self.scaler_features = scaler_features
        self.scaler_targets = scaler_targets
        
        # Initialize normalization parameters
        self.features_mean = None
        self.features_std = None
        self.targets_mean = None
        self.targets_std = None

        # If prescaler_features is None, convert it to a list of None's with length equal to number of features
        if prescaler_features is None:
            if self.request_features is not None:
                prescaler_features = [None] * len(self.request_features)
            else:
                prescaler_features = [None]
        if prescaler_targets is None:
            if self.request_targets is not None:
                prescaler_targets = [None] * len(self.request_targets)
            else:
                prescaler_targets = [None]
        
        # Setup pre-scaling functions (e.g., log transform)
        self.prescaler_features = self._setup_prescaler_functions(prescaler_features)
        self.prescaler_targets = self._setup_prescaler_functions(prescaler_targets)
    
    def _setup_prescaler_functions(self, prescaler_list):
        """Convert prescaler function names to actual numpy functions."""
        if prescaler_list is None:
            return None
        
        return [getattr(numpy, func_name) if func_name is not None else None 
                for func_name in prescaler_list]
    
    def _setup_filtering(self, filter_features, filter_targets):
        """Configure spatial filtering for features and targets."""
        self.filter_features, self.filter_features_kwargs = self._setup_filter(filter_features, "features")
        self.filter_targets, self.filter_targets_kwargs = self._setup_filter(filter_targets, "targets")
    
    def _setup_filter(self, filter_config, data_type):
        """Setup filtering configuration for features or targets."""
        if filter_config is None:
            return None, None
        
        logger.info(f"Setting up filtering for {data_type}")
        filter_config = filter_config.copy()
        
        if isinstance(filter_config, dict):
            filter_name = filter_config.pop("name", None)
            filter_func = getattr(nd, filter_name)
            filter_kwargs = filter_config
            
            # Ensure axes parameter is a tuple
            if 'axes' in filter_kwargs and isinstance(filter_kwargs['axes'], list):
                filter_kwargs['axes'] = tuple(filter_kwargs['axes'])
            
            # Validate axes for spatial filtering
            if filter_kwargs.get('axes') not in [None, (1, 2)]:
                logger.warning(
                    f"Filter axes for {data_type} should be (1,2) for spatial dimensions. "
                    f"Got: {filter_kwargs['axes']}"
                )
            
            return filter_func, filter_kwargs
        else:
            return getattr(nd, filter_config), None
    
    def _setup_transforms(self, transform):
        """Configure data augmentation transforms."""
        if transform is None:
            self.transform = None
            return
        
        transform = copy.deepcopy(transform)
        apply_to_splits = transform.pop('apply', [])
        
        if self.datalabel in apply_to_splits:
            from torchvision.transforms import v2
            logger.info(f"Applying transforms to {self.datalabel} set: {list(transform.keys())}")
            
            transform_list = [getattr(v2, name)(**params) for name, params in transform.items()]
            self.transform = v2.Compose(transform_list)
        else:
            logger.info(f"No transforms applied to {self.datalabel} set")
            self.transform = None

    
