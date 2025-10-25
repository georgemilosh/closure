
"""
dataloaders.py
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

import math
from typing import Sequence, Iterator, Optional, TypeVar
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader

import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

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