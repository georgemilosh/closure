
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
from typing import Any, Tuple, Iterator, Sequence, TypeVar, Optional
from torch.utils.data.distributed import Sampler
import torch.distributed as dist
from torch.utils.data import DataLoader
import math
import pandas as pd
import numpy as np
import joblib
import scipy.ndimage as nd

from  . import read_pic as rp

import logging
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
import copy

__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)


class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset based on provided indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it (the class works only with the indices of the dataset).

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        indices: indices of the Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices_out.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices_out to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::
        >>> sampler = DistributedSampler(indices) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """
    indices: Sequence[int]
    def __init__(self, indices: Sequence[int], num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.indices = indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.indices) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        logger.info(f"Using DistributedSampler with {self.num_replicas = }, {self.rank = }, {self.num_samples = }, {self.total_size = }, {len(self.indices) = }")

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices_out = list(self.indices[torch.randperm(len(self.indices), generator=g).tolist()])  # type: ignore[arg-type]
        else:
            indices_out = list(self.indices) #self.indices[list(range(len(self.indices)))]  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices_out)
            try:
                if padding_size <= len(indices_out):
                    indices_out += indices_out[:padding_size]
                else:
                    indices_out += (indices_out * math.ceil(padding_size / len(indices_out)))[:padding_size]
            except Exception as e:
                logger.error(f"{len(indices_out) = }, {padding_size = }, {len(indices_out[:padding_size]) = }, {len(indices_out * math.ceil(padding_size / len(indices_out))) = }")
                #logger.error(f"{indices_out = }")
                raise e
        else:
            # remove tail of data to make it evenly divisible.
            indices_out = indices_out[:self.total_size]
        assert len(indices_out) == self.total_size

        # subsample
        indices_out = indices_out[self.rank:self.total_size:self.num_replicas]
        assert len(indices_out) == self.num_samples

        return iter(indices_out)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

class SubSampler(torch.utils.data.Sampler[int]):
    """
    A custom sampler that subsamples elements from a given sequence of indices.

    Args:
        indices (Sequence[int]): The sequence of indices to subsample from.
        generator (Optional): The random number generator. Default is None.
        shuffle (bool): Whether to shuffle the indices before subsampling. Default is False.

    Returns:
        Iterator[int]: An iterator that yields the subsampled indices.

    Examples:
        >>> sampler1 = SubSampler([0,1,2,3],shuffle=True)
        >>> list(sampler1)
        [2, 0, 3, 1]
        >>> sampler1.indices
        [0, 1, 2, 3]
    """
    indices: Sequence[int]
    def __init__(self, indices: Sequence[int], seed=None, shuffle=False) -> None: #, device='cpu') -> None:
        self.indices = indices
        if seed is None:
            self.generator = None
        else:
            self.generator = torch.Generator() #device=device)
            self.generator.manual_seed(seed)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices), generator=self.generator): #, device=self.device): #  this line comes from torch.utils.data Random Sub Sampler
                yield self.indices[i]
        else:
            for i in torch.arange(len(self.indices)): #  drop randomness
                yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

class ChannelDataLoader(DataLoader):
    """
    This loader extends the PyTorch DataLoader to support distributed sampling, subsampling, and patch-based cropping.
        It allows channel-based data loading and subsampling. Channels are typically
        the different modalities or features in the dataset, such as RGB channels in an image. In case of 
        physics data they correspond to different physical quantities, such as density, pressure, etc.
        To use this loader, you need to provide a dataset that has the `request_features` and `request_targets` 
        attributes, which are lists of feature and target channel names, respectively. The loader will then sample 
        the data based on these channel names, and optionally apply subsampling and patch-based cropping. 
        Note that this data loader is designed to work with both image and vector data.
    
    See: constructor of this class for the Args.

    Attributes:
        feature_channels (list or None): The indices of the feature channels to include in the data. 
            If None, all feature channels will be included.
        target_channels (list or None): The indices of the target channels to include in the data. 
            If None, all target channels will be included.
        subsample_rate (float or None): The subsampling rate applied to the data. If None, no subsampling was applied.
        subsample_seed (int or None): The seed value used for subsampling. If None, no seed was set.

    Methods:
        __iter__(): Returns an iterator over the data, applying channel-based filtering if specified.

    Example:
        # Create a dataset
        dataset = MyDataset()

        # Create a ChannelDataLoader with specific feature and target channels
        loader = ChannelDataLoader(dataset, feature_channel_names=['channel1', 'channel2'], 
                    target_channel_names=['channel3'])

        # Iterate over the data
        for features, targets in loader:
            # Process the data
    """
    def __init__(self, dataset, feature_channel_names=None, target_channel_names=None, 
                 subsample_rate=None, subsample_seed=None, patch_dim=None, sampler_type='serial', 
                 world_size = None, rank = None, gpus_per_node = None, local_rank = None,
                 **kwargs):
        """
        Args:
            dataset (Dataset): The dataset to load the data from.
            feature_channel_names (list, optional): A list of feature channel names to include in the data. 
                features are the physical quantities (e.g. density) that are used as input to the model
                If None, all feature channels will be sampled. Default is None.
            target_channel_names (list, optional): A list of target channel names to include in the data. 
                targets are the physical quantities (e.g. pressure) that are used as output of the model.
                If None, all target channels will be sampled. Default is None.
            subsample_rate (float, optional): The subsampling rate to apply to the data, i.e. sample a 
                fraction of the dataset. If a value between 0 and 1 this is correspond to udnersampling. If above1
                then this will correspond to oversampling. Oversampling is useful when cropping is used, since cropping
                will reduce the amount of data. Oversampling in this case is crucial to be able to sample significant
                portion of the original dataset.
                If None, no subsampling will be applied. Default is None.
                Importantly subsampling creates a sampler which is used to shuffle the data. 
            subsample_seed (int, optional): The seed value for the random number generator used for subsampling. 
                If None, no seed will be set. Default is None.
            patch_dim (list, optional): The dimensions of the patch to sample from the image. Default is None.
            sampler_type (str, optional): The type of sampler to use for subsampling. Default is 'serial'. 
                Another option is 'distributed'.
            **kwargs: Additional keyword arguments to be passed to the parent DataLoader class.
        """
        self.request_features = dataset.request_features
        self.request_targets = dataset.request_targets

        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.local_rank = local_rank
        self.sampler_type = sampler_type
        
        # Convert channel names to indices
        self.feature_channels = self._get_channel_indices(feature_channel_names, self.request_features)
        self.target_channels = self._get_channel_indices(target_channel_names, self.request_targets)
        
        # Log channel selection
        logger.info(f"Selected feature channels: {self.feature_channels}")
        logger.info(f"Selected target channels: {self.target_channels}")
        
        # Configure subsampling
        self._configure_subsampling(subsample_rate, subsample_seed, dataset, kwargs)
        
        # Configure patch extraction
        self._configure_patch_extraction(patch_dim, dataset)
        
        # Create sampler and initialize parent DataLoader
        self._create_sampler_and_initialize(dataset, kwargs)
    
    def __iter__(self):
        """
        Iterate through the dataset with channel selection and patch extraction.
        
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: (features, targets) with applied transformations
        """
        for features, targets in super().__iter__():
            # Apply channel selection
            processed_features, processed_targets = self._apply_channel_selection(features, targets)
            
            # Apply patch extraction if configured
            if self.patch_dim is not None:
                processed_features, processed_targets = self._extract_patches(
                    processed_features, processed_targets
                )
            
            # Ensure proper tensor dimensions
            processed_features = self._ensure_batch_dimension(processed_features)
            processed_targets = self._ensure_batch_dimension(processed_targets)
            
            yield processed_features, processed_targets

    def _get_channel_indices(self, channel_names, available_channels):
        """Convert channel names to indices for data selection."""
        if channel_names is None:
            return None
        return [available_channels.index(channel) for channel in channel_names]

    def _configure_subsampling(self, subsample_rate, subsample_seed, dataset, kwargs):
        """Configure dataset subsampling parameters and validate batch size."""
        self.subsample_rate = subsample_rate if subsample_rate is not None else 1.0
        self.subsample_seed = subsample_seed
        
        # Set random seed for reproducibility
        if subsample_seed is not None:
            np.random.seed(subsample_seed)
        
        # Validate batch size against dataset size
        if kwargs.get('batch_size') is not None:
            max_samples = int(self.subsample_rate * len(dataset))
            batch_size = kwargs['batch_size']
            
            if batch_size > max_samples:
                raise ValueError(
                    f"Batch size ({batch_size}) must be less than dataset size × subsample_rate "
                    f"({max_samples}). Consider increasing subsample_rate or decreasing batch_size."
                )

    def _configure_patch_extraction(self, patch_dim, dataset):
        """Configure random patch extraction from images, e.g. 
        for selecting patches from images for training."""
        self.patch_dim = patch_dim
        
        if patch_dim is None:
            return
        
        # Validate patch dimensions
        if len(patch_dim) != 2:
            raise ValueError("Patch dimensions must be [width, height]")
        
        if dataset.flatten:
            raise ValueError("Patch sampling requires non-flattened data (set flatten=False)")
        
        # Validate patch size against image dimensions
        image_height, image_width = dataset.features.shape[2], dataset.features.shape[3]
        patch_width, patch_height = patch_dim
        
        if patch_width > image_width or patch_height > image_height:
            raise ValueError(
                f"Patch size ({patch_width}×{patch_height}) cannot exceed "
                f"image size ({image_width}×{image_height})"
            )
        
        logger.info(f"Patch extraction enabled: {patch_width}×{patch_height} patches")

    def _create_sampler_and_initialize(self, dataset, kwargs):
        """Create appropriate sampler and initialize parent DataLoader."""
        # Extract sampler-specific parameters
        seed = kwargs.pop('seed', None)
        shuffle = kwargs.pop('shuffle', False)
        
        # Create subset indices for subsampling
        total_samples = len(dataset.features)
        subset_size = int(total_samples * self.subsample_rate)
        
        logger.info(f"Dataset size: {total_samples} samples before subsampling")
        
        # Generate subset indices
        self.subset = np.random.permutation(subset_size)
        if self.subsample_rate > 1.0:
            # Handle oversampling by cycling through indices
            self.subset = self.subset % total_samples
        
        logger.info(f"Subset size: {len(self.subset)} samples after subsampling")
        
        # Create appropriate sampler
        if self.sampler_type == 'distributed':
            self.sampler = DistributedSampler(
                self.subset, 
                num_replicas=self.world_size, 
                rank=self.rank
            )
        elif self.sampler_type == 'serial':
            self.sampler = SubSampler(
                self.subset, 
                seed=seed, 
                shuffle=shuffle
            )
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")
        
        # Initialize parent DataLoader
        super().__init__(dataset, sampler=self.sampler, **kwargs)

   

    def _apply_channel_selection(self, features, targets):
        """Select specific channels from features and targets
             using self.feature_channels and self.target_channels."""
        if self.feature_channels is not None:
            selected_features = features[:, self.feature_channels, ...]
        else:
            selected_features = features
        
        if self.target_channels is not None:
            selected_targets = targets[:, self.target_channels, ...]
        else:
            selected_targets = targets
        
        return selected_features, selected_targets

    def _extract_patches(self, features, targets):
        """Extract random patches from features and targets.
        The method presupposes that you have run _configure_patch_extraction prior to this call."""
        batch_size = features.shape[0]
        patch_width, patch_height = self.patch_dim
        
        if len(features.shape) == 4:  # Batch of images
            # Generate random patch positions for each sample in batch
            max_x = features.shape[-2] - patch_width
            max_y = features.shape[-1] - patch_height
            
            x_starts = np.random.randint(0, max_x, size=batch_size)
            y_starts = np.random.randint(0, max_y, size=batch_size)
            
            # Pre-allocate output tensors
            patch_features = torch.empty(
                (batch_size, features.shape[1], patch_width, patch_height),
                dtype=features.dtype
            )
            patch_targets = torch.empty(
                (batch_size, targets.shape[1], patch_width, patch_height),
                dtype=targets.dtype
            )
            
            # Extract patches for each sample
            for i, (x_start, y_start) in enumerate(zip(x_starts, y_starts)):
                x_end = x_start + patch_width
                y_end = y_start + patch_height
                
                patch_features[i] = features[i, :, x_start:x_end, y_start:y_end]
                patch_targets[i] = targets[i, :, x_start:x_end, y_start:y_end]
            
            return patch_features, patch_targets
        
        elif len(features.shape) == 3:  # Single image
            max_x = features.shape[-2] - patch_width
            max_y = features.shape[-1] - patch_height
            
            x_start = np.random.randint(0, max_x)
            y_start = np.random.randint(0, max_y)
            
            x_end = x_start + patch_width
            y_end = y_start + patch_height
            
            patch_features = features[:, x_start:x_end, y_start:y_end]
            patch_targets = targets[:, x_start:x_end, y_start:y_end]
            
            return patch_features, patch_targets
        
        else:
            raise ValueError(
                f"Expected 3D or 4D tensors, got features: {features.shape}, targets: {targets.shape}"
            )

    def _ensure_batch_dimension(self, tensor):
        """Ensure tensor has batch dimension.
        If tensor is 3D, add a batch dimension at the front.
        If tensor is already 4D, return it as is."""
        if len(tensor.shape) == 3:
            return tensor.unsqueeze(0)
        return tensor

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
                 features_dtype: str = 'float32', targets_dtype: str = 'float32',
                 features_dtype_numpy: str = 'float64', targets_dtype_numpy: str = 'float64',
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
            targets_dtype (str, optional): The data type of the targets when __getitem__ is called. 
                Defaults to 'float32'.
            datalabel: Dataset split label ('train', 'val', 'test')
            flatten: If True, flatten spatial dimensions for pixel-wise processing
            image_file_name_column: Column name in CSV containing filenames
            features_dtype_numpy (str, optional): The data type of the features. Defaults to 'float32'.
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

    
