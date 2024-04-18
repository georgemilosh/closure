
"""
PyNets: A Python package for neural network training and evaluation.
Author: George Miloshevich
date: 2024
"""

import numpy
import torch
import os
from typing import Any, Optional, Callable, Tuple
import pandas as pd
import numpy as np
import joblib

import read_pic as rp

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DataFrameDataset(torch.utils.data.Dataset):
    """ This class is useful for reading dataset of images for image classification/regression problem.
    inspired by application of https://pypi.org/project/deepml/ library
    
    A dataset class for reading datasets of images for image classification/regression problems.

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
                 image_file_name_column='filenames',
                 read_features_targets_kwargs = None
                 ):
        self.target_dtype = getattr(torch, target_dtype) # parsing: convert string to torch.dtype object
        self.target_dtype_numpy = getattr(numpy, target_dtype_numpy)
        self.feature_dtype = getattr(torch, feature_dtype)
        self.feature_dtype_numpy = getattr(numpy, feature_dtype_numpy)
        self.scaler_features = scaler_features
        self.scaler_targets = scaler_targets

        self.prescaler_targets = prescaler_targets
        if prescaler_features is not None:
            self.prescaler_features = [getattr(numpy, prescaler_features) for prescaler_features in prescaler_features]
             #assuming some single variable function like numpy.log
        else:
            self.prescaler_features = prescaler_features
        
        if prescaler_targets is not None:
            self.prescaler_targets = [getattr(numpy, prescaler_targets) for prescaler_targets in prescaler_targets]
            #assuming some single variable function like numpy.log
        else:
            self.prescaler_targets = prescaler_targets

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
        self.dataframe = pd.read_csv(self.samples_file)
        self.dataframe = self.dataframe.reset_index(drop=True, inplace=False)
        
        
        filenames = self.dataframe[self.image_file_name_column].tolist()
        self.features, self.targets = rp.read_features_targets(self.data_folder, filenames, 
                                                  feature_dtype = self.feature_dtype_numpy, 
                                                  target_dtype = self.target_dtype_numpy,**self.read_features_targets_kwargs)
        self.features_shape = self.features.shape
        self.targets_shape = self.targets.shape
        self.features = self.features.reshape(-1, self.features.shape[3])
        self.targets = self.targets.reshape(-1, self.targets.shape[3])
        self.samples = self.targets.shape[0]

    def scale_data(self):    
        # === dealing with features ======
        if self.prescaler_features is not None:
            for channel in range(self.features.shape[1]):
                self.features[:,channel,...] = self.prescaler_features[channel](self.features[:,channel,...])
                logger.info(f"Prescaling { self.prescaler_features[channel]} applied to features")
        if self.scaler_features is not None:
            processing_folder, samples_file_name = self.samples_file.rsplit('/', 1)
            name = f'{self.norm_folder}/X_{samples_file_name}_{str(self.prescaler_features)}.pkl'
            if isinstance(self.scaler_features, tuple):
                logger.info(f"dataset provided with scaler features")
                self.features_mean, self.features_std = self.scaler_features
            else:
                if os.path.exists(name):
                    self.features_mean, self.features_std = joblib.load(name)
                    logger.info(f"Loaded self.features_mean, self.features_std from {name}")
                else:
                    self.features_mean = np.asarray(np.mean(self.features, axis=0), dtype=self.feature_dtype_numpy)
                    self.features_std = np.asarray(np.std(self.features, axis=0), dtype=self.feature_dtype_numpy)
                    joblib.dump((self.features_mean, self.features_std), name)
                    logger.info(f"Saved self.features_mean, self.features_std to {name}")
            logger.info("Normalization applied to features")
            for channel in range(self.features.shape[1]):
                self.features[:,channel,...] -= self.features_mean[channel]
                self.features[:,channel,...] /= self.features_std[channel]
        # === dealing with targets ======
        if self.prescaler_targets is not None:
            for channel in range(self.targets.shape[1]):
                self.targets[:,channel,...] = self.prescaler_targets[channel](self.targets[:,channel,...])
                logger.info(f"Prescaling { self.prescaler_targets[channel]} applied to targets")    
        if self.scaler_targets is not None:
            processing_folder, samples_file_name = self.samples_file.rsplit('/', 1)
            name = f'{self.norm_folder}/y_{samples_file_name}_{str(self.prescaler_targets)}.pkl'
            if isinstance(self.scaler_targets, tuple):
                logger.info(f"dataset provided with scaler targets")
                self.targets_mean, self.targets_std = self.scaler_targets
            else:
                if os.path.exists(name):
                    self.targets_mean, self.targets_std = joblib.load(name)
                    logger.info(f"Loaded self.targets_mean, self.targets_std from {name}")
                else:
                    self.targets_mean = np.asarray(np.mean(self.targets, axis=0), dtype=self.target_dtype_numpy)
                    self.targets_std = np.asarray(np.std(self.targets, axis=0), dtype=self.target_dtype_numpy)
                    joblib.dump((self.targets_mean, self.targets_std), name)
                    logger.info(f"Saved self.targets_mean, self.targets_std to {name}")
            logger.info("Normalization applied to targets")
            self.targets -= self.targets_mean
            self.targets /= self.targets_std


    def __len__(self):
        return self.samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        features, targets = self.features[idx], self.targets[idx]
        features = torch.tensor(features, dtype=self.feature_dtype)
        targets = torch.tensor(targets, dtype=self.target_dtype)
        #if self.transform is not None:
        #    features = self.transform(features)
        #if self.target_transform is not None:
        #    targets = self.target_transform(targets)
        return features, targets
