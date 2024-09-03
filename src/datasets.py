
"""
PyNets: A Python package for neural network training and evaluation.
Author: George Miloshevich
date: 2024
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

        >>> # xdoctest: +SKIP
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

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices_out = torch.randperm(len(self.indices), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices_out = list(range(len(self.indices)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices_out)
            if padding_size <= len(indices_out):
                indices_out += indices_out[:padding_size]
            else:
                indices_out += (indices_out * math.ceil(padding_size / len(indices_out)))[:padding_size]
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
        #self.device = device
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
    A custom data loader that allows for channel-based data loading and subsampling.

    Args:
        dataset (Dataset): The dataset to load the data from.
        feature_channel_names (list, optional): A list of feature channel names to include in the data. If None, all feature channels will be sampled. Default is None.
        target_channel_names (list, optional): A list of target channel names to include in the data. If None, all target channels will be sampled. Default is None.
        subsample_rate (float, optional): The subsampling rate to apply to the data. Should be a value between 0 and 1. If None, no subsampling will be applied. Default is None.
        Importantly subsampling creates a sampler which is used to shuffle the data. 
        subsample_seed (int, optional): The seed value for the random number generator used for subsampling. If None, no seed will be set. Default is None.
        patch_dim (list, optional): The dimensions of the patch to sample from the image. Default is None.
        sampler_type (str, optional): The type of sampler to use for subsampling. Default is 'serial'. Another option is 'distributed'.
        **kwargs: Additional keyword arguments to be passed to the parent DataLoader class.

    Attributes:
        feature_channels (list or None): The indices of the feature channels to include in the data. If None, all feature channels will be included.
        target_channels (list or None): The indices of the target channels to include in the data. If None, all target channels will be included.
        subsample_rate (float or None): The subsampling rate applied to the data. If None, no subsampling was applied.
        subsample_seed (int or None): The seed value used for subsampling. If None, no seed was set.

    Methods:
        __iter__(): Returns an iterator over the data, applying channel-based filtering if specified.

    Example:
        # Create a dataset
        dataset = MyDataset()

        # Create a ChannelDataLoader with specific feature and target channels
        loader = ChannelDataLoader(dataset, feature_channel_names=['channel1', 'channel2'], target_channel_names=['channel3'])

        # Iterate over the data
        for features, targets in loader:
            # Process the data
    """
    def __init__(self, dataset, feature_channel_names=None, target_channel_names=None, 
                 subsample_rate=None, subsample_seed=None, patch_dim=None, sampler_type='serial', 
                 world_size = None, rank = None, gpus_per_node = None, local_rank = None,
                 **kwargs):
        self.request_features = dataset.request_features
        self.request_targets = dataset.request_targets
        self.sampler_type = sampler_type
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.local_rank = local_rank
        if feature_channel_names is not None:
            self.feature_channels = [self.request_features.index(channel) for channel in feature_channel_names]
        else:
            self.feature_channels = None
        if target_channel_names is not None:
            self.target_channels = [self.request_targets.index(channel) for channel in target_channel_names]
        else:
            self.target_channels = None
        logger.info(f"ChannelDataLoader.feature_channels: {self.feature_channels}")
        logger.info(f"ChannelDataLoader.target_channels: {self.target_channels}")
        
        self.subsample_rate = subsample_rate
        self.subsample_seed = subsample_seed
        if self.subsample_rate is not None:
            assert kwargs['batch_size'] <= self.subsample_rate*len(dataset), "Batch size must be less than the number of samples in the dataset times subsample rate (ideally several times). Try increasing the latter"
        else:
            assert kwargs['batch_size'] <= len(dataset), "Batch size must be less than the number of samples in the dataset"
        if self.subsample_seed is not None:
            np.random.seed(self.subsample_seed)
        
        self.patch_dim = patch_dim
        if self.patch_dim is not None:
            logger.info(f"Using {self.patch_dim = }")
            assert len(self.patch_dim) == 2, "Patch dimensions must be a list of length 2"
            assert dataset.flatten == False, "Patch sampling only works with non-flattened data"
            assert self.patch_dim[0] <= dataset.features.shape[2], "Patch width must be less than or equal to the image width"
            assert self.patch_dim[1] <= dataset.features.shape[3], "Patch height must be less than or equal to the image height"

        seed = kwargs.pop('seed', None) # these are only needed if subsample_rate is not None
        shuffle = kwargs.pop('shuffle', False) # these are only needed if subsample_rate is not None

        if self.subsample_rate is None:
            self.subsample_rate = 1

        
        logger.info(f"{len(dataset.features)}, {len(dataset.targets) = } samples before subsampling")
        self.subset = np.random.permutation(int(len(dataset.features)*self.subsample_rate))
        if self.subsample_rate > 1:
            self.subset = self.subset % len(dataset.features)  # if subsample_rate > 1, then we want to loop over the dataset multiple times
            
        logger.info(f"{len(self.subset) = } samples after subsampling")
        
        if sampler_type == 'distributed':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.subset,
                                                                    num_replicas=self.world_size,
                                                                    rank=self.rank)
        elif sampler_type == 'serial':
            self.sampler = SubSampler(self.subset, seed=seed, shuffle=shuffle)
        else:
            raise ValueError(f"Sampler type {sampler_type} not recognized")
            #super().__init__(dataset, sampler=self.sampler, **kwargs)
        #else:
        #    logger.info(f"Using full dataset (no subsampling)")
            #super().__init__(dataset, **kwargs) # normal operation without specifying subsampling
        super().__init__(dataset, sampler=self.sampler, **kwargs)
    
    def __iter__(self):
        for features, targets in super().__iter__():
            if self.feature_channels is not None or self.target_channels is not None:
                patched_features = features[:, self.feature_channels, ...]
                patched_targets = targets[:, self.target_channels, ...]
            else:
                patched_features = features
                patched_targets = targets

            if self.patch_dim is not None:
                # Generate random start points for patches
                x_starts = np.random.randint(0, features.shape[2] - self.patch_dim[0], size=features.shape[0])
                y_starts = np.random.randint(0, features.shape[3] - self.patch_dim[1], size=features.shape[0])
                patched_features = torch.empty((features.shape[0], features.shape[1], self.patch_dim[0], self.patch_dim[1]))
                patched_targets = torch.empty((targets.shape[0], targets.shape[1], self.patch_dim[0], self.patch_dim[1]))
                
                # Extract patches using advanced indexing
                for i, (x, y) in enumerate(zip(x_starts, y_starts)):
                    patched_features[i] = features[i, :, x:x+self.patch_dim[0], y:y+self.patch_dim[1]]
                    patched_targets[i] = targets[i, :, x:x+self.patch_dim[0], y:y+self.patch_dim[1]]
            
            yield patched_features, patched_targets

class DataFrameDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch dataset class for loading data from a DataFrame.

    Args:
        data_folder (str): The folder where the images are stored.
        norm_folder (str): The folder to save the normalization parameters.
        feature_dtype (str, optional): The data type of the features when __getitem__ is called. Defaults to 'float32'.
        target_dtype (str, optional): The data type of the targets when __getitem__ is called. Defaults to 'float32'.
        feature_dtype_numpy (str, optional): The data type of the features. Defaults to 'float32'.
        target_dtype_numpy (str, optional): The data type of the targets. Defaults to 'float32'.
        samples_file (str, optional): The file containing the sample filenames. Defaults to None.
        prescaler_features (str, optional): The pre-scaler function to apply to the features. Defaults to None.
        prescaler_targets (str, optional): The pre-scaler function to apply to the targets. Defaults to None.
        scaler_features (tuple or None, optional): The scaler for features. If a tuple is provided, it should contain the mean and standard deviation of the features. Defaults to None.
        scaler_targets (tuple or None, optional): The scaler for targets. If a tuple is provided, it should contain the mean and standard deviation of the targets. Defaults to None.
        image_file_name_column (str, optional): The column name in the DataFrame that contains the image filenames. Defaults to 'filenames'.
        read_features_targets_kwargs (dict, optional): Additional keyword arguments to pass to the `read_features_targets` function. Defaults to None.
        filter_features (str, optional): The filter to apply to the features. Defaults to None.
        filter_targets (str, optional): The filter to apply to the targets. Defaults to None.

    Attributes:
        target_dtype (torch.dtype): The data type of the targets.
        target_dtype_numpy (numpy.dtype): The data type of the targets in numpy format.
        feature_dtype (torch.dtype): The data type of the features.
        feature_dtype_numpy (numpy.dtype): The data type of the features in numpy format.
        scaler_features (tuple or None): The scaler for features.
        scaler_targets (tuple or None): The scaler for targets.
        prescaler_features (list or None): The pre-scaler functions to apply to the features.
        prescaler_targets (list or None): The pre-scaler functions to apply to the targets.
        samples_file (str or None): The file containing the sample filenames.
        image_file_name_column (str): The column name in the DataFrame that contains the image filenames.
        data_folder (str): The folder where the images are stored.
        norm_folder (str): The folder to save the normalization parameters.
        read_features_targets_kwargs (dict): Additional keyword arguments to pass to the `read_features_targets` function.
        logger: The logger object for logging messages.
        dataframe (pd.DataFrame): The DataFrame containing the sample filenames.
        flatten (bool): Whether to flatten the features and targets. Default is True.
        features (np.ndarray): The features of the dataset.
        targets (np.ndarray): The targets of the dataset.
        features_shape (tuple): The shape of the features.
        targets_shape (tuple): The shape of the targets.
        samples (int): The number of samples in the dataset.
        request_features (list or None): The requested features to load.
        request_targets (list or None): The requested targets to load.

    Methods:
        load_original(): Loads the DataFrame from a CSV file and prepares the features and targets for further processing.
        scale_data(): Scales the features and targets of the dataset using pre-defined scalers or calculates and saves new scalers if necessary.
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the features and targets for a given index.

    """
    def __init__(self, data_folder: str,  # where the images are stored
                 norm_folder: str, # where to save the normalization parameters
                 feature_dtype = 'float32', # data type of the features when __getitem__ is called
                 target_dtype = 'float32', # data type of the targets when __getitem__ is called
                 feature_dtype_numpy = 'float32', # data type of the features
                 target_dtype_numpy = 'float32', # data type of the targets
                 samples_file = None,
                 prescaler_features: str = None,
                 prescaler_targets: str = None,
                 scaler_features = None,
                 scaler_targets = None,
                 datalabel = 'train',
                 flatten = True,
                 image_file_name_column='filenames',
                 read_features_targets_kwargs = None,
                 filter_features = None, # filter to apply to the features
                filter_targets = None, # filter to apply to the targets
                 ):
        self.features_mean = None  # TODO: check that this doesn't break something
        self.features_std = None
        self.flatten = flatten
        
        self.target_dtype = getattr(torch, target_dtype) # parsing: convert string to torch.dtype object
        self.target_dtype_numpy = getattr(numpy, target_dtype_numpy)
        self.feature_dtype = getattr(torch, feature_dtype)
        self.feature_dtype_numpy = getattr(numpy, feature_dtype_numpy)
        self.scaler_features = scaler_features
        self.scaler_targets = scaler_targets

        self.prescaler_targets = prescaler_targets
        if prescaler_features is not None:
            self.prescaler_features = [getattr(numpy, prescaler_features) 
                                       if prescaler_features is not None else None for 
                                       prescaler_features in prescaler_features] #assuming some single variable function like numpy.log
        else:
            self.prescaler_features = prescaler_features
        
        self.filter_featuers_kwargs = None
        
        if filter_features is not None:
            filter_features_copy = filter_features.copy()
            logger.info("Filtering features")
            if not isinstance(filter_features, dict):
                self.filter_features = getattr(nd, filter_features_copy)
            else:
                filter_features_name = filter_features_copy.pop("name", None)
                self.filter_features = getattr(nd, filter_features_name)
                self.filter_featuers_kwargs = filter_features_copy
                if isinstance(self.filter_featuers_kwargs['axes'], list):
                    self.filter_featuers_kwargs['axes'] = tuple(self.filter_featuers_kwargs['axes'])
                if self.filter_featuers_kwargs['axes'] is None or self.filter_featuers_kwargs['axes'] != (1,2):
                    logger.warning(f"Filtering features should be aplied to only spatial dimensions. {self.filter_featuers_kwargs['axes'] = } and it should be (1.2)")
        else:
            self.filter_features = None
        self.filter_targets_kwargs = None
        if filter_targets is not None:
            filter_targets_copy = filter_targets.copy()
            logger.info("Filtering targets")
            if not isinstance(filter_targets, dict):
                self.filter_targets = getattr(nd, filter_targets_copy)
            else:
                filter_targets_name = filter_targets_copy.pop("name", None)
                self.filter_targets = getattr(nd, filter_targets_name)
                self.filter_targets_kwargs = filter_targets_copy
                if isinstance(self.filter_targets_kwargs['axes'], list):
                    self.filter_targets_kwargs['axes'] = tuple(self.filter_targets_kwargs['axes'])
                if self.filter_targets_kwargs['axes'] is None or self.filter_targets_kwargs['axes'] != (1,2):
                    logger.warning("Filtering targets should be aplied to only spatial dimensions")
        else:
            self.filter_targets = None

        if prescaler_targets is not None:
            self.prescaler_targets = [getattr(numpy, prescaler_targets) 
                                      if prescaler_targets is not None else None for 
                                      prescaler_targets in prescaler_targets] #assuming some single variable function like numpy.log
        else:
            self.prescaler_targets = prescaler_targets
        self.datalabel = datalabel
        logger.info(f" This is {self.datalabel} set")
        self.samples_file = samples_file
        self.image_file_name_column = image_file_name_column
        self.data_folder = data_folder
        self.norm_folder = norm_folder
        if read_features_targets_kwargs is None:
            self.read_features_targets_kwargs = {}
        else:
            self.read_features_targets_kwargs = read_features_targets_kwargs

        self.logger = logger

        self.load_original()
        self.scale_data()
    
    def load_original(self):
        """
        Loads the atarame from a  file containing filenames which will be used to create this dataset. 
        Prepares the features and targets for further processing based on this dataframe.

        """
        logger.info(f"Datasplit performed according to {self.samples_file}")
        self.dataframe = pd.read_csv(self.samples_file)
        self.dataframe = self.dataframe.reset_index(drop=True, inplace=False)
        
        
        self.filenames = self.dataframe[self.image_file_name_column].tolist()

        self.request_features = self.read_features_targets_kwargs.get('request_features', None)
        self.request_targets = self.read_features_targets_kwargs.get('request_targets', None)
        self.features, self.targets = rp.read_features_targets(self.data_folder, self.filenames, 
                                                  feature_dtype = self.feature_dtype_numpy, 
                                                  target_dtype = self.target_dtype_numpy,**self.read_features_targets_kwargs)
        if self.filter_features is not None:
            self.features = self.filter_features(self.features, **self.filter_featuers_kwargs)
        if self.filter_targets is not None:
            self.targets = self.filter_targets(self.targets, **self.filter_targets_kwargs)
        self.features_shape = self.features.shape
        self.targets_shape = self.targets.shape
        if self.flatten:
            self.features = self.features.reshape(-1, self.features.shape[-1])
            self.targets = self.targets.reshape(-1, self.targets.shape[-1])
        else:
            self.features = self.features.transpose(0, 3, 1, 2)
            self.targets = self.targets.transpose(0, 3, 1, 2)
        logger.info(f"Features shape: {self.features.shape}, Targets shape: {self.targets.shape}")

        self.samples = self.targets.shape[0]

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
        # === dealing with features ======
        if self.prescaler_features is not None and self.prescaler_features is not False:
            for channel in range(self.features.shape[1]):
                if self.prescaler_features[channel] is not None:
                    self.features[:,channel,...] = self.prescaler_features[channel](self.features[:,channel,...])
                    logger.info(f"Prescaling { self.prescaler_features[channel]} applied to features")
        if self.scaler_features is not None and self.scaler_features is not False:
            #processing_folder, samples_file_name = self.samples_file.rsplit('/', 1)
            name = f'{self.norm_folder}/X.pkl' #_{samples_file_name}_{str(self.prescaler_features)}.pkl' # X_{samples_file_name}_{str(self.prescaler_features)}.pkl'
            if isinstance(self.scaler_features, tuple):
                logger.info(f"dataset provided with scaler features") # TODO: check if self.datalabel is correct
                self.features_mean, self.features_std = self.scaler_features
            else:
                if os.path.exists(name):
                    self.features_mean, self.features_std = joblib.load(name)
                    logger.info(f"Loaded self.features_mean, self.features_std from {name}")
                else:
                    if len(self.features.shape) > 2:
                        self.features_mean = np.asarray(np.mean(self.features, axis=(0, 2, 3)), dtype=self.feature_dtype_numpy)
                        self.features_std = np.asarray(np.std(self.features, axis=(0, 2, 3)), dtype=self.feature_dtype_numpy)
                    else:
                        self.features_mean = np.asarray(np.mean(self.features, axis=0), dtype=self.feature_dtype_numpy)
                        self.features_std = np.asarray(np.std(self.features, axis=0), dtype=self.feature_dtype_numpy)
                    joblib.dump((self.features_mean, self.features_std), name)
                    logger.info(f"Saved self.features_mean, self.features_std to {name}")
            logger.info("Normalization applied to features")
            for channel in range(self.features.shape[1]):
                try:
                    self.features[:,channel,...] -= self.features_mean[channel]
                except Exception as e:
                    logger.info(f"{self.features.shape = }, {self.features_mean = }")
                    raise e
                self.features[:,channel,...] /= self.features_std[channel]
        else:
            self.features_mean = None
            self.features_std = None
        # === dealing with targets ======
        if self.prescaler_targets is not None and self.prescaler_targets is not False:
            for channel in range(self.targets.shape[1]):
                if self.prescaler_targets[channel] is not None:
                    self.targets[:,channel,...] = self.prescaler_targets[channel](self.targets[:,channel,...])
                    logger.info(f"Prescaling { self.prescaler_targets[channel]} applied to targets")    
        if self.scaler_targets is not None and self.scaler_targets is not False:
            #processing_folder, samples_file_name = self.samples_file.rsplit('/', 1)
            name = f'{self.norm_folder}/y.pkl' #y_{samples_file_name}_{str(self.prescaler_targets)}.pkl'
            if isinstance(self.scaler_targets, tuple):
                logger.info(f"dataset provided with scaler targets")
                self.targets_mean, self.targets_std = self.scaler_targets
            else:
                if os.path.exists(name):
                    self.targets_mean, self.targets_std = joblib.load(name)
                    logger.info(f"Loaded self.targets_mean, self.targets_std from {name}")
                else:
                    if len(self.targets.shape) > 2:
                        self.targets_mean = np.asarray(np.mean(self.targets, axis=(0, 2, 3)), dtype=self.target_dtype_numpy)
                        self.targets_std = np.asarray(np.std(self.targets, axis=(0, 2, 3)), dtype=self.target_dtype_numpy)
                    else:
                        self.targets_mean = np.asarray(np.mean(self.targets, axis=0), dtype=self.target_dtype_numpy)
                        self.targets_std = np.asarray(np.std(self.targets, axis=0), dtype=self.target_dtype_numpy)
                    joblib.dump((self.targets_mean, self.targets_std), name)
                    logger.info(f"Saved self.targets_mean, self.targets_std to {name}")
            logger.info("Normalization applied to targets")
            for channel in range(self.targets.shape[1]):
                self.targets[:,channel,...] -= self.targets_mean[channel]
                self.targets[:,channel,...] /= self.targets_std[channel]
        else:
            self.targets_mean = None
            self.targets_std = None
        self.features = torch.tensor(self.features, dtype=self.feature_dtype)
        self.targets = torch.tensor(self.targets, dtype=self.target_dtype)    


    def __len__(self):
        return self.samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Tells data loaders how to load the data from the dataset.
        """        
        features, targets = self.features[idx], self.targets[idx]
        #features = torch.tensor(features, dtype=self.feature_dtype)
        #targets = torch.tensor(targets, dtype=self.target_dtype)
        #if self.transform is not None:
        #    features = self.transform(features)
        #if self.target_transform is not None:
        #    targets = self.target_transform(targets)
        return features, targets
