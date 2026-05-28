
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

import bisect as _bisect
import concurrent.futures as _cf
import hashlib as _hashlib
import inspect
import json as _json
import numpy
try:
    import torch
except ImportError:
    print("datasets: PyTorch is not installed. Some functions may not work.")
import os
from collections import OrderedDict
from typing import Any, Iterator, List, Optional, Tuple, TypeVar

try:
    from tqdm import tqdm as _tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False


def _progress(iterable=None, **kwargs):
    """Wrap iterable with tqdm if available, otherwise return it unchanged.

    When called with no iterable (context-manager pattern for manual updates),
    returns a tqdm bar or a no-op context manager.
    """
    if iterable is None:
        return _tqdm(**kwargs) if _has_tqdm else _NullContext()
    return _tqdm(iterable, **kwargs) if _has_tqdm else iterable


class _NullContext:
    """No-op context manager used when tqdm is unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def update(self, n=1):
        pass

import pandas as pd
import numpy as np
import joblib
import scipy.ndimage as nd


from  . import read_pic as rp

import logging
logger = logging.getLogger(__name__)


def _dbg_mem(tag: str, extra: str = "") -> None:
    """Log RSS for the current process when CLOSURE_DEBUG_MEM=1.

    Cheap no-op when disabled. Used to attribute RAM growth to specific
    preprocessing / chunk-load events so OOMs can be traced to a source.
    """
    import os as _os
    if _os.environ.get("CLOSURE_DEBUG_MEM", "") != "1":
        return
    try:
        import psutil as _psutil
        rss_gb = _psutil.Process().memory_info().rss / 1024 ** 3
    except Exception:
        rss_gb = float("nan")
    wid = _os.environ.get("PYTORCH_DATALOADER_WORKER_ID", "main")
    try:
        import torch as _torch
        info = _torch.utils.data.get_worker_info()
        if info is not None:
            wid = str(info.id)
    except Exception:
        pass
    logger.info("[DBGMEM] pid=%d worker=%s rss=%.2fGB %s %s",
                _os.getpid(), wid, rss_gb, tag, extra)

__all__ = [
    "DataFrameDataset",
    "LazyNPZDataFrameDataset",
    "OnePatchPerFileBatchSampler",
    "FileChunkedSampler",
    "PreprocessedChunkDataset",
    "ChunkOrderedSampler",
]


import copy

T_co = TypeVar('T_co', covariant=True)


class _Compose:
    """Lightweight transform compose to avoid torchvision dependency."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class _RandomCrop:
    """Random crop for CHW tensors using torch RNG state."""

    def __init__(self, size):
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(f"RandomCrop size must be a 2-item list/tuple, got: {size}")
        self.crop_h = int(size[0])
        self.crop_w = int(size[1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"RandomCrop expects CHW tensor, got shape {tuple(x.shape)}")

        _, height, width = x.shape
        if self.crop_h > height or self.crop_w > width:
            raise ValueError(
                f"Crop size {(self.crop_h, self.crop_w)} exceeds input spatial shape {(height, width)}"
            )

        max_top = height - self.crop_h
        max_left = width - self.crop_w
        top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,)).item())
        left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,)).item())
        return x[:, top:top + self.crop_h, left:left + self.crop_w]


_LOCAL_TRANSFORMS = {
    "RandomCrop": _RandomCrop,
}

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
                     transform: dict = None,
                     alfven_units: bool = False):
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
                alfven_units (bool, optional): If True, rescale each sample from code
                    units to Alfvén units using the ``.inp`` file auto-detected from
                    its experiment subdirectory. Defaults to False.
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
            self.alfven_units = alfven_units
            self.alfven_params: dict[str, dict[str, float]] = {}
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
        
        
        # Keep only kwargs accepted by read_pic.read_features_targets to avoid
        # breaking when notebook/config carries stale keys from older APIs.
        valid_params = set(inspect.signature(rp.read_features_targets).parameters.keys())
        filtered_kwargs = {
            k: v for k, v in self.read_features_targets_kwargs.items() if k in valid_params
        }
        dropped_kwargs = sorted(
            k for k in self.read_features_targets_kwargs.keys() if k not in valid_params
        )
        if dropped_kwargs:
            logger.warning(
                "Ignoring unsupported read_features_targets kwargs: %s",
                dropped_kwargs,
            )

        # Load features and targets from files
        self.features, self.targets = rp.read_features_targets(
            self.data_folder, self.filenames,
            features_dtype=self.features_dtype_numpy,
            targets_dtype=self.targets_dtype_numpy,
            alfven_units=self.alfven_units,
            **filtered_kwargs
        )

        # Build Alfvén parameter cache for downstream use (visualization, inference)
        if self.alfven_units:
            from .plasma import (
                _find_experiment_inp_file,
                _read_b0x_nb_from_inp,
                alfven_scales,
            )
            for fn in self.filenames:
                exp_dir = rp._resolve_experiment_dir(self.data_folder, fn)
                if exp_dir not in self.alfven_params:
                    b0x, nb = _read_b0x_nb_from_inp(_find_experiment_inp_file(exp_dir))
                    self.alfven_params[exp_dir] = alfven_scales(b0x, nb)
        
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
        # Accumulate in float64: float32 sums saturate past ~10^7 elements
        # per channel, producing wrong mean/std on production-scale splits.
        if len(data.shape) > 2:
            mean = np.mean(data, axis=(0, 2, 3), dtype=np.float64).astype(dtype_numpy)
            std = np.std(data, axis=(0, 2, 3), dtype=np.float64).astype(dtype_numpy)
        else:
            mean = np.mean(data, axis=0, dtype=np.float64).astype(dtype_numpy)
            std = np.std(data, axis=0, dtype=np.float64).astype(dtype_numpy)
        
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
            logger.info(f"Applying transforms to {self.datalabel} set: {list(transform.keys())}")

            transform_list = []
            for name, params in transform.items():
                if name not in _LOCAL_TRANSFORMS:
                    raise ValueError(
                        f"Unsupported transform '{name}'. Supported transforms without torchvision: "
                        f"{sorted(_LOCAL_TRANSFORMS.keys())}"
                    )
                transform_list.append(_LOCAL_TRANSFORMS[name](**params))

            self.transform = _Compose(transform_list)
        else:
            logger.info(f"No transforms applied to {self.datalabel} set")
            self.transform = None


class _NPZFastPathUnavailable(Exception):
    """Signal from the npz fast path that a per-file constraint is violated."""


class _FileLRUCache:
    """Tiny in-process LRU cache holding decoded per-file arrays.

    Keys are file indices; values are ``(features, targets)`` numpy tuples
    after prescaling and (optional) normalization.  When ``capacity == 0`` the
    cache is a no-op.  One instance lives per ``LazyNPZDataFrameDataset``
    object; in a multi-worker ``DataLoader`` each worker process gets its own
    independent cache.
    """

    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self._store: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: int):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: int, value: Tuple[np.ndarray, np.ndarray]) -> None:
        if self.capacity == 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


class LazyNPZDataFrameDataset(DataFrameDataset):
    """Disk-backed dataset that keeps only metadata and global stats in RAM.

    Indexing matches the eager ``DataFrameDataset``:

    * ``flatten=False`` (CNN/FCNN with ``patch_dim``): one sample per file,
      ``__len__ == num_files``.  ``__getitem__(file_idx)`` returns ``(C, H, W)``
      tensors and ``self.transform`` (e.g. ``RandomCrop``) runs per access.
    * ``flatten=True`` (MLP, pixel-wise): one sample per pixel,
      ``__len__ == num_files * H * W``.  ``__getitem__(global_idx)`` decodes
      ``file_idx = global_idx // (H*W)`` and ``pixel_idx = global_idx % (H*W)``,
      then returns a single ``(C,)`` feature/target vector.

    A small per-worker LRU cache keyed by ``file_idx`` keeps decoded files in
    memory across consecutive accesses so that pixel-wise sampling or
    oversampling does not pay decode cost per pixel/crop.  Pair with the
    samplers below (``OnePatchPerFileBatchSampler`` for ``flatten=False``,
    ``FileChunkedSampler`` for ``flatten=True``) to keep cache hit-rate high
    while preserving decorrelation across batches.

    Normalization statistics remain global over the training split.  When
    missing, train datasets stream once over every training file (bypassing
    the cache) and save the usual ``X.pkl`` / ``y.pkl`` files for all splits
    to reuse.
    """

    def __init__(
        self,
        data_folder: str,
        norm_folder: str,
        samples_file: str = None,
        features_dtype: str = 'float32',
        feature_dtype: str = None,
        targets_dtype: str = 'float32',
        target_dtype: str = None,
        features_dtype_numpy: str = 'float64',
        feature_dtype_numpy: str = None,
        targets_dtype_numpy: str = 'float64',
        target_dtype_numpy: str = None,
        prescaler_features: list = None,
        prescaler_targets: list = None,
        scaler_features: bool = None,
        scaler_targets: bool = None,
        datalabel: str = 'train',
        flatten: bool = True,
        image_file_name_column: str = 'filenames',
        read_features_targets_kwargs: dict = None,
        filter_features: dict = None,
        filter_targets: dict = None,
        transform: dict = None,
        alfven_units: bool = False,
        sample_cache_size: int = 1,
    ):
        if feature_dtype is not None:
            features_dtype = feature_dtype
        if feature_dtype_numpy is not None:
            features_dtype_numpy = feature_dtype_numpy
        if target_dtype is not None:
            targets_dtype = target_dtype
        if target_dtype_numpy is not None:
            targets_dtype_numpy = target_dtype_numpy

        self.data_folder = data_folder
        self.norm_folder = norm_folder
        self.samples_file = samples_file
        self.datalabel = datalabel
        self.flatten = flatten
        self.image_file_name_column = image_file_name_column
        self.alfven_units = alfven_units
        self.alfven_params: dict[str, dict[str, float]] = {}
        self.logger = logger
        self.read_features_targets_kwargs = read_features_targets_kwargs or {}
        self.request_features = self.read_features_targets_kwargs.get('request_features', None)
        self.request_targets = self.read_features_targets_kwargs.get('request_targets', None)

        logger.info(f" This is lazy {self.datalabel} set")

        self._setup_data_types(features_dtype, targets_dtype, features_dtype_numpy, targets_dtype_numpy)
        self._setup_preprocessing(prescaler_features, prescaler_targets, scaler_features, scaler_targets)
        self._setup_filtering(filter_features, filter_targets)
        self._setup_transforms(transform)

        self.dataframe = pd.read_csv(self.samples_file).reset_index(drop=True)
        self.filenames = self.dataframe[self.image_file_name_column].tolist()
        self.num_files = len(self.filenames)
        self._read_kwargs = self._filtered_read_features_targets_kwargs()

        if self.num_files == 0:
            raise ValueError(f"No samples found in {self.samples_file}")

        # Per-instance (and therefore per-DataLoader-worker) LRU cache holding
        # decoded, prescaled, normalized arrays in CHW layout.
        self.sample_cache_size = int(sample_cache_size)
        self._cache = _FileLRUCache(self.sample_cache_size)

        # Tri-state flag for the single-open .npz fast path:
        #   None  -> not yet probed; check on first load.
        #   True  -> fast path validated; use it for every file.
        #   False -> incompatible; permanently fall back to rp.read_features_targets.
        self._npz_fast_path: Optional[bool] = None

        # Probe shapes from the first file (CHW layout, before any flatten).
        features_chw, targets_chw = self._load_file_chw(0, normalize=False)
        c_f, h, w = features_chw.shape
        c_t, h_t, w_t = targets_chw.shape
        if (h, w) != (h_t, w_t):
            raise ValueError(
                f"feature/target spatial shapes disagree: {(h, w)} vs {(h_t, w_t)}"
            )
        self._h = int(h)
        self._w = int(w)
        self._pixels_per_file = self._h * self._w

        # ``features_shape`` mirrors eager ``DataFrameDataset`` pre-reshape
        # layout: ``(N_files, H, W, C)``.  Channel-name resolution downstream
        # only reads ``self.request_features``/``self.request_targets``.
        self.features_shape = (self.num_files, self._h, self._w, c_f)
        self.targets_shape = (self.num_files, self._h, self._w, c_t)

        # ``self.samples`` follows eager semantics: pixel count for flatten,
        # file count otherwise.  Used by ``__len__`` and ``_maybe_subsample``.
        if self.flatten:
            self.samples = self.num_files * self._pixels_per_file
        else:
            self.samples = self.num_files

        self._prepare_normalization_params("features")
        self._prepare_normalization_params("targets")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if self.flatten:
            file_idx, pixel_idx = divmod(int(idx), self._pixels_per_file)
            features_chw, targets_chw = self._get_file_arrays(file_idx, normalize=True)
            # CHW -> (C, H*W); pick one column.
            features_vec = features_chw.reshape(features_chw.shape[0], -1)[:, pixel_idx]
            targets_vec = targets_chw.reshape(targets_chw.shape[0], -1)[:, pixel_idx]
            return (
                torch.as_tensor(features_vec, dtype=self.features_dtype),
                torch.as_tensor(targets_vec, dtype=self.targets_dtype),
            )

        features_chw, targets_chw = self._get_file_arrays(int(idx), normalize=True)
        features = torch.as_tensor(features_chw, dtype=self.features_dtype)
        targets = torch.as_tensor(targets_chw, dtype=self.targets_dtype)
        if self.transform is not None:
            state = torch.get_rng_state()
            features = self.transform(features)
            torch.set_rng_state(state)
            targets = self.transform(targets)
        return features, targets

    # ------------------------------------------------------------------
    # File loading (cached)
    # ------------------------------------------------------------------
    def _get_file_arrays(
        self, file_idx: int, normalize: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(features, targets)`` in CHW layout for ``file_idx``.

        Cache holds only the normalized variant (the common training path).
        Stats computation calls with ``normalize=False`` and skips the cache.
        """
        if normalize:
            cached = self._cache.get(file_idx)
            if cached is not None:
                return cached
        features, targets = self._load_file_chw(file_idx, normalize=normalize)
        if normalize:
            self._cache.put(file_idx, (features, targets))
        return features, targets

    def _load_file_chw(
        self, file_idx: int, normalize: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read one file from disk and return ``(features, targets)`` as CHW."""
        filename = self.filenames[file_idx]

        # Fast path: when the file is a flat ``.npz`` whose keys directly
        # match ``request_features`` / ``request_targets`` and no Alfvén
        # rescaling / spatial slicing / filter callback is configured, we
        # open the npz once and pull every requested field in a single pass
        # instead of letting ``rp.read_features_targets`` re-open the file
        # once per channel (16x I/O for the 10F/6T production setup).
        features, targets = self._maybe_fast_load_npz(filename)
        if features is None:
            features, targets = rp.read_features_targets(
                self.data_folder,
                [filename],
                features_dtype=self.features_dtype_numpy,
                targets_dtype=self.targets_dtype_numpy,
                alfven_units=self.alfven_units,
                **self._read_kwargs,
            )

            if self.filter_features is not None:
                features = self.filter_features(features, **self.filter_features_kwargs)
            if self.filter_targets is not None:
                targets = self.filter_targets(targets, **self.filter_targets_kwargs)

            # rp.read_features_targets returns (N=1, H, W, C); convert to (C, H, W).
            features = features.transpose(0, 3, 1, 2)[0]
            targets = targets.transpose(0, 3, 1, 2)[0]

        self._apply_prescaling_to_sample(features, self.prescaler_features, "features")
        self._apply_prescaling_to_sample(targets, self.prescaler_targets, "targets")

        if normalize:
            features = self._normalize_sample(features, "features")
            targets = self._normalize_sample(targets, "targets")

        return features, targets

    # ------------------------------------------------------------------
    # Single-open .npz fast path
    # ------------------------------------------------------------------
    def _maybe_fast_load_npz(
        self, filename: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(features_chw, targets_chw)`` via a single ``np.load`` or ``(None, None)``.

        Returns ``(None, None)`` to signal the caller should use the standard
        ``rp.read_features_targets`` path. Once a file is found to be
        incompatible (missing keys, slicing requested, etc.) the fast path is
        permanently disabled for the dataset instance.
        """
        if getattr(self, "_npz_fast_path", None) is False:
            return None, None
        if not filename.endswith(".npz"):
            # Per-file format check; do not poison the dataset-level flag.
            return None, None
        # Dataset-level invariants: if any of these are set, the fast path
        # can never apply; disable permanently so we don't re-check.
        if (
            self.alfven_units
            or self.filter_features is not None
            or self.filter_targets is not None
            or not self.request_features
            or not self.request_targets
        ):
            self._npz_fast_path = False
            return None, None
        rk = self._read_kwargs
        if any(rk.get(k) is not None for k in ("choose_x", "choose_y", "choose_z")):
            self._npz_fast_path = False
            return None, None
        # Derived-field markers (nested species lists) must go through rp.
        for fld in (*self.request_features, *self.request_targets):
            if not isinstance(fld, str):
                self._npz_fast_path = False
                return None, None

        path = os.path.join(self.data_folder, filename)
        try:
            with np.load(path) as npz:
                available = set(npz.files)
                missing = [
                    k for k in (*self.request_features, *self.request_targets)
                    if k not in available
                ]
                if missing:
                    if self._npz_fast_path is None:
                        logger.info(
                            "npz fast path disabled: %s missing keys %s; "
                            "falling back to read_features_targets",
                            filename, missing,
                        )
                    self._npz_fast_path = False
                    return None, None
                feat_arrs = [np.asarray(npz[k]) for k in self.request_features]
                targ_arrs = [np.asarray(npz[k]) for k in self.request_targets]
        except Exception as exc:
            if self._npz_fast_path is None:
                logger.info(
                    "npz fast path disabled for %s (%s); falling back",
                    filename, exc,
                )
            self._npz_fast_path = False
            return None, None

        try:
            features = self._stack_npz_arrays(feat_arrs, self.features_dtype_numpy)
            targets = self._stack_npz_arrays(targ_arrs, self.targets_dtype_numpy)
        except _NPZFastPathUnavailable as exc:
            if self._npz_fast_path is None:
                logger.info(
                    "npz fast path disabled for %s (%s); falling back",
                    filename, exc,
                )
            self._npz_fast_path = False
            return None, None

        if self._npz_fast_path is None:
            self._npz_fast_path = True
            logger.info("npz fast path enabled for %s", self.data_folder)
        return features, targets

    @staticmethod
    def _stack_npz_arrays(arrs: List[np.ndarray], dtype) -> np.ndarray:
        """Stack ``[(z,y,x) | (y,x)]`` channel arrays into ``(C, H, W)``.

        Reproduces what ``rp.read_fieldname`` does for ``.npz`` inputs when
        ``choose_x/y/z`` default to ``None``:

        * Each axis with size > 1 is sliced ``[0:size-1]`` (rp off-by-one);
        * The per-channel result is transposed from ``(y, x)`` to
          ``(x, y)`` ordering (rp ``indexing='ij'`` transpose).

        Both quirks must be preserved for the fast path to be a drop-in
        substitute; see ``read_pic.read_fieldname``.
        """

        def _slice_and_swap(plane_yx: np.ndarray) -> np.ndarray:
            h, w = plane_yx.shape
            sliced = plane_yx[: h - 1 if h > 1 else 1, : w - 1 if w > 1 else 1]
            # (y, x) -> (x, y) to match rp's transpose(2, 1, 0) on (z, y, x).
            return np.ascontiguousarray(sliced.T)

        def _to_plane(a: np.ndarray) -> np.ndarray:
            if a.ndim == 3:
                if a.shape[0] != 1:
                    raise _NPZFastPathUnavailable(
                        f"unexpected z-dim {a.shape[0]}; need z=1"
                    )
                return a[0]
            if a.ndim == 2:
                return a
            raise _NPZFastPathUnavailable(f"unexpected ndim={a.ndim}")

        sample = _slice_and_swap(_to_plane(arrs[0]))
        h, w = sample.shape
        out = np.empty((len(arrs), h, w), dtype=dtype)
        for c, a in enumerate(arrs):
            plane = _slice_and_swap(_to_plane(a))
            if plane.shape != (h, w):
                raise _NPZFastPathUnavailable(
                    f"channel {c} sliced shape {plane.shape} != ({h}, {w})"
                )
            out[c] = plane.astype(dtype, copy=False)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _filtered_read_features_targets_kwargs(self) -> dict:
        valid_params = set(inspect.signature(rp.read_features_targets).parameters.keys())
        filtered_kwargs = {
            k: v for k, v in self.read_features_targets_kwargs.items() if k in valid_params
        }
        dropped_kwargs = sorted(
            k for k in self.read_features_targets_kwargs.keys() if k not in valid_params
        )
        if dropped_kwargs:
            logger.warning(
                "Ignoring unsupported read_features_targets kwargs: %s",
                dropped_kwargs,
            )
        return filtered_kwargs

    def _apply_prescaling_to_sample(self, data, prescaler_functions, data_type):
        if prescaler_functions is None:
            return
        for channel in range(data.shape[0]):
            if prescaler_functions[channel] is not None:
                data[channel, ...] = prescaler_functions[channel](data[channel, ...])

    def _normalize_sample(self, data, data_type):
        if not getattr(self, f'scaler_{data_type}'):
            return data
        mean = getattr(self, f'{data_type}_mean')
        std = getattr(self, f'{data_type}_std')
        shape = (mean.shape[0],) + (1,) * (data.ndim - 1)
        return (data - mean.reshape(shape)) / std.reshape(shape)

    def _prepare_normalization_params(self, data_type):
        if not getattr(self, f'scaler_{data_type}'):
            return

        filename = self._normalization_filename(data_type)

        # DDP coordination: only rank 0 computes & writes; other ranks wait on
        # a barrier and then load the file from disk. Falls back to the normal
        # single-process path when ``torch.distributed`` is not initialized.
        ddp_active = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if ddp_active else 0

        if os.path.exists(filename):
            mean, std = joblib.load(filename)
            logger.info(f"Loaded normalization parameters for {data_type} from {filename}")
        elif self.datalabel == 'train':
            if rank == 0:
                mean, std = self._compute_streaming_normalization_params(data_type)
                os.makedirs(self.norm_folder, exist_ok=True)
                joblib.dump((mean, std), filename)
                logger.info(
                    f"Computed and saved lazy normalization parameters for "
                    f"{data_type} to {filename}"
                )
            if ddp_active:
                torch.distributed.barrier()
            if rank != 0:
                mean, std = joblib.load(filename)
                logger.info(
                    f"Rank {rank} loaded normalization parameters for "
                    f"{data_type} from {filename}"
                )
        else:
            raise ValueError(
                f"Normalization parameters for {data_type} not found at {filename}. "
                "Parameters must be computed on training data first."
            )

        setattr(self, f'{data_type}_mean', mean)
        setattr(self, f'{data_type}_std', std)

    def _normalization_filename(self, data_type):
        return f'{self.norm_folder}/{"X" if data_type == "features" else "y"}.pkl'

    def _compute_streaming_normalization_params(self, data_type):
        """One streaming pass over every training file in float64.

        Equivalent to the eager per-channel mean/std over the whole train
        split: reduces over (file, height, width) for every channel.  Bypasses
        the LRU cache to keep RAM flat.
        """
        dtype_numpy = getattr(self, f'{data_type}_dtype_numpy')
        total: Optional[np.ndarray] = None
        total_sq: Optional[np.ndarray] = None
        count = 0

        for file_idx in range(self.num_files):
            features, targets = self._load_file_chw(file_idx, normalize=False)
            data = features if data_type == "features" else targets
            flat = data.reshape(data.shape[0], -1).astype(np.float64, copy=False)
            sample_sum = flat.sum(axis=1)
            sample_sum_sq = np.square(flat).sum(axis=1)
            if total is None:
                total = sample_sum
                total_sq = sample_sum_sq
            else:
                total += sample_sum
                total_sq += sample_sum_sq
            count += flat.shape[1]

        mean = (total / count).astype(dtype_numpy)
        variance = total_sq / count - np.square(total / count)
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance).astype(dtype_numpy)
        return mean, std


# ----------------------------------------------------------------------
# Samplers for lazy NPZ datasets
# ----------------------------------------------------------------------
class OnePatchPerFileBatchSampler(torch.utils.data.Sampler):
    """Batch sampler yielding ``batch_size`` distinct file indices per batch.

    Designed for ``LazyNPZDataFrameDataset`` with ``flatten=False`` and a
    ``RandomCrop`` transform.  Guarantees that no two samples in a batch come
    from the same file (i.e. the same time snapshot), eliminating within-batch
    time correlation while keeping a tiny per-worker file cache effective.

    Parameters
    ----------
    num_files
        Number of distinct files in the dataset (``dataset.num_files``).
    batch_size
        Batch size.  Must be ``<= num_files``.
    oversample
        Number of passes through the shuffled file deck per epoch.  Mirrors the
        legacy ``subsample_rate`` oversampling on top of ``patch_dim`` random
        crops: each file is visited ``oversample`` times, yielding a different
        crop each time.
    drop_last
        If True, drop the final incomplete batch in each pass.
    shuffle
        If False, file order is the deterministic ``range(num_files)``.
    seed
        Base seed; the epoch index is added to it via :meth:`set_epoch`.
    """

    def __init__(
        self,
        num_files: int,
        batch_size: int,
        *,
        oversample: int = 1,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if num_files <= 0:
            raise ValueError(f"num_files must be positive, got {num_files}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > num_files:
            raise ValueError(
                f"batch_size {batch_size} > num_files {num_files} is incompatible "
                "with one-patch-per-file sampling (a batch would need duplicate files)"
            )
        self.num_files = int(num_files)
        self.batch_size = int(batch_size)
        self.oversample = max(1, int(oversample))
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        for _ in range(self.oversample):
            if self.shuffle:
                order = torch.randperm(self.num_files, generator=g).tolist()
            else:
                order = list(range(self.num_files))
            n_full = (len(order) // self.batch_size) * self.batch_size
            for i in range(0, n_full, self.batch_size):
                yield order[i:i + self.batch_size]
            if not self.drop_last and n_full < len(order):
                yield order[n_full:]

    def __len__(self) -> int:
        per_pass = self.num_files // self.batch_size
        if not self.drop_last and self.num_files % self.batch_size:
            per_pass += 1
        return per_pass * self.oversample


class FileChunkedSampler(torch.utils.data.Sampler):
    """Pixel-index sampler that processes files in small windows.

    Designed for ``LazyNPZDataFrameDataset`` with ``flatten=True``.  Yields
    global pixel indices ``file_idx * pixels_per_file + pixel_idx`` such that,
    within each window of ``window`` consecutive files, all pixels of those
    files are emitted (in random order) before moving on.  This keeps a per-
    worker LRU cache of size ``window`` warm while still randomizing pixel and
    file order across the epoch.

    For ``window == 1`` each file is fully consumed before the next is opened
    (best I/O, weakest decorrelation).  Larger ``window`` improves across-file
    interleaving within a batch at the cost of holding ``window`` files in the
    cache simultaneously.

    Parameters
    ----------
    num_files
        Number of distinct files.
    pixels_per_file
        ``H * W`` (after any spatial filtering).
    window
        Number of files held in flight together.
    shuffle
        If False, fixed order; useful for val/test reproducibility.
    seed
        Base seed; epoch index is added via :meth:`set_epoch`.
    """

    def __init__(
        self,
        num_files: int,
        pixels_per_file: int,
        *,
        window: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if num_files <= 0:
            raise ValueError(f"num_files must be positive, got {num_files}")
        if pixels_per_file <= 0:
            raise ValueError(f"pixels_per_file must be positive, got {pixels_per_file}")
        self.num_files = int(num_files)
        self.pixels_per_file = int(pixels_per_file)
        self.window = max(1, int(window))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        if self.shuffle:
            file_order = torch.randperm(self.num_files, generator=g).tolist()
        else:
            file_order = list(range(self.num_files))

        ppf = self.pixels_per_file
        for start in range(0, self.num_files, self.window):
            group = file_order[start:start + self.window]
            if self.shuffle:
                pixel_orders = [
                    torch.randperm(ppf, generator=g).tolist() for _ in group
                ]
            else:
                pixel_orders = [list(range(ppf)) for _ in group]
            # Round-robin across files in the window so adjacent yielded
            # indices come from different files — this is what gives the
            # decorrelation while still bounding the live cache at ``window``.
            for p_idx in range(ppf):
                for f_local, f_idx in enumerate(group):
                    yield f_idx * ppf + pixel_orders[f_local][p_idx]

    def __len__(self) -> int:
        return self.num_files * self.pixels_per_file


# -----------------------------------------------------------------------
# Preprocessed / chunked SSD-backed dataset
# -----------------------------------------------------------------------

def _available_ram_bytes() -> int:
    """Query available system RAM from /proc/meminfo, falling back to 16 GiB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 16 * 1024 ** 3


def _files_per_chunk(
    C_f: int, C_t: int, H: int, W: int,
    num_gpus: int = 1,
    preprocess_chunk_size_gb: float = None,
    safety: float = 0.4,
) -> int:
    """Number of simulation files that fit in one preprocessed chunk."""
    bytes_per_file = max(1, (C_f + C_t) * H * W * 4)  # float32
    if preprocess_chunk_size_gb is not None:
        budget = int(preprocess_chunk_size_gb * 1024 ** 3)
    else:
        budget = int(_available_ram_bytes() * safety / max(1, num_gpus))
    return max(1, budget // bytes_per_file)


def _preprocessing_fingerprint(
    samples_file: str,
    request_features,
    request_targets,
    prescaler_features,
    prescaler_targets,
    alfven_units: bool,
) -> str:
    """Short hash of the preprocessing config for cache-directory naming."""

    def _fname(f):
        if f is None:
            return 'none'
        return f.__name__ if callable(f) else str(f)

    data = {
        "sf": str(samples_file),
        "rf": [str(x) for x in (request_features or [])],
        "rt": [str(x) for x in (request_targets or [])],
        "pf": [_fname(f) for f in (prescaler_features or [])],
        "pt": [_fname(f) for f in (prescaler_targets or [])],
        "au": bool(alfven_units),
    }
    return _hashlib.md5(_json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]


class _ChunkLRUCache:
    """LRU cache holding loaded chunk tensor pairs keyed by chunk index."""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._store: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

    def get(self, key: int):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: int, value: "Tuple[torch.Tensor, torch.Tensor]") -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)


class PreprocessedChunkDataset(torch.utils.data.Dataset):
    """Preprocessing-once, SSD-backed dataset for fast training.

    At first use (or when the cache is absent), streams raw simulation
    files through :class:`LazyNPZDataFrameDataset` — prescaling + global
    normalization — groups them into RAM-bounded chunks, and writes
    ``float32`` tensors to ``ssd_cache_dir``.  Subsequent training reads
    pre-normalised tensors directly from SSD with no per-batch processing.

    Each chunk file (``chunk_NNNN.pt``) is a ``torch.save`` dict::

        {"features": Tensor(N, C_f, H, W), "targets": Tensor(N, C_t, H, W)}

    A fingerprinted sub-directory is used so different experiment
    configurations never share a cache.

    Parameters
    ----------
    ssd_cache_dir : str
        Local SSD directory.  Chunk files land in
        ``ssd_cache_dir/{datalabel}_{fingerprint}/``.
    chunk_cache_size : int
        Chunk tensors kept resident per DataLoader worker.  Default ``1``.
    preprocess_chunk_size_gb : float | None
        RAM budget (GiB) per chunk during the preprocessing pass.
        Auto-estimated from ``/proc/meminfo`` / ``num_gpus`` when ``None``.
    num_gpus : int
        Number of GPUs; divides the available-RAM budget for chunk sizing.
    preprocess_num_workers : int
        Number of threads used to read files in parallel during the
        preprocessing pass.  Default ``1`` (sequential).  Increasing this
        speeds up the initial preprocessing at the cost of slightly higher
        peak RAM (``preprocess_num_workers`` files in flight at once).
    """

    def __init__(
        self,
        data_folder: str,
        norm_folder: str,
        samples_file: str,
        ssd_cache_dir: str,
        features_dtype: str = 'float32',
        feature_dtype: str = None,
        targets_dtype: str = 'float32',
        target_dtype: str = None,
        features_dtype_numpy: str = 'float64',
        feature_dtype_numpy: str = None,
        targets_dtype_numpy: str = 'float64',
        target_dtype_numpy: str = None,
        prescaler_features: list = None,
        prescaler_targets: list = None,
        scaler_features: bool = None,
        scaler_targets: bool = None,
        datalabel: str = 'train',
        flatten: bool = True,
        image_file_name_column: str = 'filenames',
        read_features_targets_kwargs: dict = None,
        filter_features: dict = None,
        filter_targets: dict = None,
        transform: dict = None,
        alfven_units: bool = False,
        chunk_cache_size: int = 1,
        preprocess_chunk_size_gb: float = None,
        num_gpus: int = 1,
        preprocess_num_workers: int = 1,
    ):
        if feature_dtype is not None:
            features_dtype = feature_dtype
        if feature_dtype_numpy is not None:
            features_dtype_numpy = feature_dtype_numpy
        if target_dtype is not None:
            targets_dtype = target_dtype
        if target_dtype_numpy is not None:
            targets_dtype_numpy = target_dtype_numpy

        self.data_folder = data_folder
        self.norm_folder = norm_folder
        self.samples_file = samples_file
        self.datalabel = datalabel
        self.flatten = flatten
        self.image_file_name_column = image_file_name_column
        self.alfven_units = alfven_units
        self.alfven_params: dict = {}
        self.logger = logger

        self.read_features_targets_kwargs = read_features_targets_kwargs or {}
        self.request_features = self.read_features_targets_kwargs.get('request_features', None)
        self.request_targets = self.read_features_targets_kwargs.get('request_targets', None)

        self.features_dtype = getattr(torch, features_dtype)
        self.targets_dtype = getattr(torch, targets_dtype)
        self.features_dtype_numpy = getattr(numpy, features_dtype_numpy)
        self.targets_dtype_numpy = getattr(numpy, targets_dtype_numpy)

        self.scaler_features = scaler_features
        self.scaler_targets = scaler_targets

        # Convert prescaler name strings → numpy callables (for logging compat).
        def _to_funcs(lst, n):
            if lst is None:
                return [None] * n
            return [
                getattr(numpy, f) if (f is not None and not callable(f)) else f
                for f in lst
            ]

        n_f = len(self.request_features or [])
        n_t = len(self.request_targets or [])
        self.prescaler_features = _to_funcs(prescaler_features, n_f)
        self.prescaler_targets = _to_funcs(prescaler_targets, n_t)

        # Keep raw forms for passing to LazyNPZDataFrameDataset internals.
        self._prescaler_features_raw = prescaler_features
        self._prescaler_targets_raw = prescaler_targets
        self._filter_features_raw = filter_features
        self._filter_targets_raw = filter_targets
        self._preprocess_chunk_size_gb = preprocess_chunk_size_gb
        self._num_gpus = num_gpus
        self._preprocess_num_workers = max(1, int(preprocess_num_workers))

        # Transform (e.g. RandomCrop) applied at __getitem__ time.
        self._setup_transforms(transform)

        # Fingerprint-based cache directory (collision-free across configs).
        fp = _preprocessing_fingerprint(
            samples_file, self.request_features, self.request_targets,
            self.prescaler_features, self.prescaler_targets, alfven_units,
        )
        self._chunk_dir = os.path.join(ssd_cache_dir, f"{datalabel}_{fp}")
        self._meta_path = os.path.join(self._chunk_dir, "metadata.json")

        # DDP coordination: rank 0 preprocesses, others wait at the barrier.
        ddp_active = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if ddp_active else 0

        if not os.path.exists(self._meta_path):
            if rank == 0:
                self._preprocess_and_save()
            if ddp_active:
                torch.distributed.barrier()
        elif ddp_active:
            torch.distributed.barrier()

        # Load metadata written by the preprocessing pass.
        with open(self._meta_path) as f:
            meta = _json.load(f)

        self.num_files = meta['num_files']
        self._h = meta['H']
        self._w = meta['W']
        self._c_f = meta['C_f']
        self._c_t = meta['C_t']
        self._pixels_per_file = self._h * self._w
        self._chunk_sizes = meta['chunk_sizes']
        self._num_chunks = len(self._chunk_sizes)
        # file_perm[slot] = CSV file index stored at that slot.
        # Absent in caches written before this feature was added (treated as identity).
        self._file_perm: List[int] = meta.get('file_perm', list(range(self.num_files)))

        # Cumulative file offsets: chunk i covers files [offsets[i], offsets[i+1]).
        self._chunk_offsets = [0]
        for s in self._chunk_sizes:
            self._chunk_offsets.append(self._chunk_offsets[-1] + s)

        self.features_shape = (self.num_files, self._h, self._w, self._c_f)
        self.targets_shape = (self.num_files, self._h, self._w, self._c_t)
        self.samples = (
            self.num_files * self._pixels_per_file if self.flatten else self.num_files
        )

        # Load norm stats for attribute compatibility (data already normalized).
        self.features_mean = self.features_std = None
        self.targets_mean = self.targets_std = None
        for dt in ('features', 'targets'):
            fname = f'{norm_folder}/{"X" if dt == "features" else "y"}.pkl'
            if getattr(self, f'scaler_{dt}') and os.path.exists(fname):
                mean, std = joblib.load(fname)
                setattr(self, f'{dt}_mean', mean)
                setattr(self, f'{dt}_std', std)

        # Per-worker LRU cache for loaded chunk tensors.
        self._chunk_cache = _ChunkLRUCache(chunk_cache_size)

        logger.info(
            "PreprocessedChunkDataset | split=%s | files=%d | chunks=%d | "
            "flatten=%s | samples=%d",
            datalabel, self.num_files, self._num_chunks, flatten, self.samples,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int):
        if self.flatten:
            file_idx, pixel_idx = divmod(int(idx), self._pixels_per_file)
        else:
            file_idx = int(idx)
            pixel_idx = None

        chunk_id, local_idx = self._file_to_chunk(file_idx)
        feat_chunk, targ_chunk = self._load_chunk(chunk_id)

        features = feat_chunk[local_idx]  # (C_f, H, W)
        targets = targ_chunk[local_idx]   # (C_t, H, W)

        if self.flatten:
            return (
                features.reshape(self._c_f, -1)[:, pixel_idx].to(self.features_dtype),
                targets.reshape(self._c_t, -1)[:, pixel_idx].to(self.targets_dtype),
            )

        features = features.to(self.features_dtype)
        targets = targets.to(self.targets_dtype)
        if self.transform is not None:
            state = torch.get_rng_state()
            features = self.transform(features)
            torch.set_rng_state(state)
            targets = self.transform(targets)
        return features, targets

    # ------------------------------------------------------------------
    # Chunk access
    # ------------------------------------------------------------------

    def _file_to_chunk(self, file_idx: int) -> "Tuple[int, int]":
        """Return (chunk_id, local_file_index_within_chunk)."""
        chunk_id = _bisect.bisect_right(self._chunk_offsets, file_idx) - 1
        chunk_id = min(max(chunk_id, 0), self._num_chunks - 1)
        return chunk_id, file_idx - self._chunk_offsets[chunk_id]

    def _load_chunk(self, chunk_id: int) -> "Tuple[torch.Tensor, torch.Tensor]":
        cached = self._chunk_cache.get(chunk_id)
        if cached is not None:
            return cached
        path = os.path.join(self._chunk_dir, f"chunk_{chunk_id:04d}.pt")
        data = torch.load(path, map_location='cpu', weights_only=True)
        result = (data['features'], data['targets'])
        self._chunk_cache.put(chunk_id, result)
        bytes_loaded = result[0].element_size() * result[0].numel() + \
                       result[1].element_size() * result[1].numel()
        _dbg_mem(
            "chunk_load",
            f"chunk_id={chunk_id} bytes={bytes_loaded/1024**3:.2f}GB "
            f"cache_occ={len(self._chunk_cache._store)}/{self._chunk_cache.capacity}",
        )
        return result

    # ------------------------------------------------------------------
    # Preprocessing (rank 0 only)
    # ------------------------------------------------------------------

    def _preprocess_and_save(self) -> None:
        """Two-pass preprocessing: stream norm stats → normalize → save chunks.

        Called only on DDP rank 0 (or in single-process runs).  Uses
        :class:`LazyNPZDataFrameDataset` with ``scaler=False`` for the first
        pass so no DDP barrier is triggered inside it.  After saving the norm
        pickles manually, the second pass creates a normalized
        ``LazyNPZDataFrameDataset`` that simply loads from the files we just
        wrote — again no barrier.
        """
        logger.info(
            "PreprocessedChunkDataset: preprocessing %s split → %s",
            self.datalabel, self._chunk_dir,
        )
        os.makedirs(self._chunk_dir, exist_ok=True)
        _dbg_mem("preprocess_start", f"split={self.datalabel}")

        dtype_f = numpy.dtype(self.features_dtype_numpy).name
        dtype_t = numpy.dtype(self.targets_dtype_numpy).name

        # Pass 1: raw reader (prescaling on, normalization off) for stat streaming.
        raw = LazyNPZDataFrameDataset(
            data_folder=self.data_folder,
            norm_folder=self.norm_folder,
            samples_file=self.samples_file,
            features_dtype_numpy=dtype_f,
            targets_dtype_numpy=dtype_t,
            prescaler_features=self._prescaler_features_raw,
            prescaler_targets=self._prescaler_targets_raw,
            scaler_features=False,
            scaler_targets=False,
            datalabel=self.datalabel,
            flatten=False,
            image_file_name_column=self.image_file_name_column,
            read_features_targets_kwargs=self.read_features_targets_kwargs,
            filter_features=self._filter_features_raw,
            filter_targets=self._filter_targets_raw,
            alfven_units=self.alfven_units,
            sample_cache_size=1,
        )

        num_files = raw.num_files
        H, W = raw._h, raw._w
        C_f = raw.features_shape[3]
        C_t = raw.targets_shape[3]

        # Compute and persist norm stats before creating the normalized reader.
        self._stream_norm_stats(raw)

        # Pass 2: normalized reader (loads stats from pkl, no DDP barrier).
        normalized = LazyNPZDataFrameDataset(
            data_folder=self.data_folder,
            norm_folder=self.norm_folder,
            samples_file=self.samples_file,
            features_dtype_numpy=dtype_f,
            targets_dtype_numpy=dtype_t,
            prescaler_features=self._prescaler_features_raw,
            prescaler_targets=self._prescaler_targets_raw,
            scaler_features=self.scaler_features,
            scaler_targets=self.scaler_targets,
            datalabel=self.datalabel,
            flatten=False,
            image_file_name_column=self.image_file_name_column,
            read_features_targets_kwargs=self.read_features_targets_kwargs,
            filter_features=self._filter_features_raw,
            filter_targets=self._filter_targets_raw,
            alfven_units=self.alfven_units,
            sample_cache_size=1,
        )

        fpb = _files_per_chunk(C_f, C_t, H, W, self._num_gpus, self._preprocess_chunk_size_gb)
        logger.info(
            "Chunk size: %d files/chunk  (C_f=%d C_t=%d H=%d W=%d "
            "→ %.1f MiB/chunk)",
            fpb, C_f, C_t, H, W,
            fpb * (C_f + C_t) * H * W * 4 / 1024 ** 2,
        )

        # Shuffle file order before chunking so each chunk is a random mix
        # of files rather than a consecutive block from the CSV (which may be
        # time-ordered).  A fixed seed makes the layout deterministic for cache
        # reuse; the sampler re-shuffles within each chunk every epoch anyway.
        file_perm = numpy.random.RandomState(0).permutation(num_files).tolist()

        # Pre-compute per-chunk file-index lists.
        chunks_file_indices: List[List[int]] = []
        start = 0
        while start < num_files:
            end = min(start + fpb, num_files)
            chunks_file_indices.append([file_perm[s] for s in range(start, end)])
            start = end

        def _read_file(file_idx):
            feat_chw, targ_chw = normalized._load_file_chw(file_idx, normalize=True)
            return (
                torch.from_numpy(numpy.ascontiguousarray(feat_chw)).float(),
                torch.from_numpy(numpy.ascontiguousarray(targ_chw)).float(),
            )

        chunk_sizes: List[int] = []
        with _progress(total=num_files, desc="preprocess", unit="file", leave=True) as pbar:
            with _cf.ThreadPoolExecutor(max_workers=self._preprocess_num_workers) as pool:
                for chunk_idx, file_indices in enumerate(chunks_file_indices):
                    futures = [pool.submit(_read_file, fi) for fi in file_indices]
                    buf_f = []
                    buf_t = []
                    for fut in futures:
                        feat_t, targ_t = fut.result()
                        buf_f.append(feat_t)
                        buf_t.append(targ_t)
                        pbar.update(1)

                    chunk_path = os.path.join(self._chunk_dir, f"chunk_{chunk_idx:04d}.pt")
                    torch.save(
                        {
                            "features": torch.stack(buf_f, dim=0),
                            "targets": torch.stack(buf_t, dim=0),
                        },
                        chunk_path,
                    )
                    chunk_sizes.append(len(buf_f))
                    logger.info(
                        "Saved chunk %d (%d files) → %s",
                        chunk_idx, len(buf_f), chunk_path,
                    )
                    _dbg_mem("preprocess_chunk_saved",
                             f"chunk_idx={chunk_idx} files={len(buf_f)}")

        meta = {
            "version": 1,
            "num_files": num_files,
            "num_chunks": len(chunk_sizes),
            "chunk_sizes": chunk_sizes,
            "H": H,
            "W": W,
            "C_f": C_f,
            "C_t": C_t,
            "file_perm": file_perm,
        }
        with open(self._meta_path, 'w') as fh:
            _json.dump(meta, fh, indent=2)
        logger.info(
            "Preprocessing complete: %d files → %d chunks in %s",
            num_files, len(chunk_sizes), self._chunk_dir,
        )

    def _stream_norm_stats(self, raw: "LazyNPZDataFrameDataset") -> None:
        """Streaming pass to compute and save per-channel mean/std.

        Uses a thread pool when ``preprocess_num_workers > 1`` so multiple
        files are read concurrently while stats are accumulated serially.
        """
        for data_type in ('features', 'targets'):
            if not getattr(self, f'scaler_{data_type}'):
                continue
            fname = f'{self.norm_folder}/{"X" if data_type == "features" else "y"}.pkl'
            if os.path.exists(fname):
                logger.info("Norm stats for %s exist at %s, reusing", data_type, fname)
                continue

            dtype_numpy = getattr(self, f'{data_type}_dtype_numpy')
            total: Optional[numpy.ndarray] = None
            total_sq: Optional[numpy.ndarray] = None
            count = 0

            def _file_partial(file_idx, _dt=data_type):
                feat, targ = raw._load_file_chw(file_idx, normalize=False)
                arr = feat if _dt == 'features' else targ
                flat = arr.reshape(arr.shape[0], -1).astype(numpy.float64, copy=False)
                return flat.sum(axis=1), numpy.square(flat).sum(axis=1), flat.shape[1]

            if self._preprocess_num_workers > 1:
                with _cf.ThreadPoolExecutor(max_workers=self._preprocess_num_workers) as pool:
                    futs = {pool.submit(_file_partial, i): i for i in range(raw.num_files)}
                    for fut in _progress(
                        _cf.as_completed(futs),
                        total=raw.num_files,
                        desc=f"norm stats ({data_type})",
                        unit="file",
                        leave=False,
                    ):
                        t, t_sq, c = fut.result()
                        total = t if total is None else total + t
                        total_sq = t_sq if total_sq is None else total_sq + t_sq
                        count += c
            else:
                for file_idx in _progress(
                    range(raw.num_files),
                    desc=f"norm stats ({data_type})",
                    unit="file",
                    leave=False,
                ):
                    t, t_sq, c = _file_partial(file_idx)
                    total = t if total is None else total + t
                    total_sq = t_sq if total_sq is None else total_sq + t_sq
                    count += c

            mean = (total / count).astype(dtype_numpy)
            variance = numpy.maximum(total_sq / count - numpy.square(total / count), 0.0)
            std = numpy.sqrt(variance).astype(dtype_numpy)
            os.makedirs(self.norm_folder, exist_ok=True)
            joblib.dump((mean, std), fname)
            logger.info("Saved norm stats for %s → %s", data_type, fname)

    def _setup_transforms(self, transform) -> None:
        if transform is None:
            self.transform = None
            return
        transform = copy.deepcopy(transform)
        apply_to_splits = transform.pop('apply', [])
        if self.datalabel in apply_to_splits:
            transform_list = []
            for name, params in transform.items():
                if name not in _LOCAL_TRANSFORMS:
                    raise ValueError(
                        f"Unsupported transform '{name}'. "
                        f"Supported: {sorted(_LOCAL_TRANSFORMS.keys())}"
                    )
                transform_list.append(_LOCAL_TRANSFORMS[name](**params))
            self.transform = _Compose(transform_list)
        else:
            self.transform = None


class ChunkOrderedSampler(torch.utils.data.Sampler):
    """Index sampler that visits chunks sequentially to minimise cache thrashing.

    Designed for :class:`PreprocessedChunkDataset`.  Each epoch:

    1. Chunk order is shuffled.
    2. Within each chunk all local indices are shuffled.
    3. Indices are yielded chunk-by-chunk so a per-worker
       :class:`_ChunkLRUCache` of size 1 never evicts inside a chunk.

    For ``flatten=False`` (CNN/FCNN), set ``pixels_per_file=1`` and use
    ``oversample`` to visit each file multiple times per epoch (each visit
    applies a different :class:`_RandomCrop` patch).

    For ``flatten=True`` (MLP), set ``pixels_per_file = H * W``; ``oversample``
    is ignored (all pixels are already emitted once per epoch).

    Parameters
    ----------
    chunk_sizes : list[int]
        Number of files in each chunk (``dataset._chunk_sizes``).
    pixels_per_file : int
        ``1`` for image mode, ``H * W`` for pixel mode.
    oversample : int
        Passes through the full file deck per epoch (image mode only).
    shuffle : bool
        If False the order is deterministic (useful for val/test).
    seed : int
        Base seed; the epoch index is added via :meth:`set_epoch`.
    """

    def __init__(
        self,
        chunk_sizes: List[int],
        pixels_per_file: int = 1,
        *,
        oversample: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not chunk_sizes:
            raise ValueError("chunk_sizes must be non-empty")
        self.chunk_sizes = list(chunk_sizes)
        self.pixels_per_file = max(1, int(pixels_per_file))
        self.oversample = max(1, int(oversample))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

        # Cumulative file offsets for global index computation.
        self._chunk_offsets = [0]
        for s in self.chunk_sizes:
            self._chunk_offsets.append(self._chunk_offsets[-1] + s)
        self._total_files = self._chunk_offsets[-1]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._total_files * self.pixels_per_file * self.oversample

    def __iter__(self) -> Iterator[int]:
        ppf = self.pixels_per_file
        for pass_idx in range(self.oversample):
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch * self.oversample + pass_idx)

            if self.shuffle:
                chunk_order = torch.randperm(len(self.chunk_sizes), generator=g).tolist()
            else:
                chunk_order = list(range(len(self.chunk_sizes)))

            for chunk_id in chunk_order:
                n_files = self.chunk_sizes[chunk_id]
                base = self._chunk_offsets[chunk_id] * ppf
                total_in_chunk = n_files * ppf

                if self.shuffle:
                    local_perm = torch.randperm(total_in_chunk, generator=g).tolist()
                else:
                    local_perm = list(range(total_in_chunk))

                for local_idx in local_perm:
                    yield base + local_idx
