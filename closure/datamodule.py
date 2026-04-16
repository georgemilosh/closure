"""
datamodule.py — ClosureDataModule: PyTorch Lightning data module for closure.

Wraps :class:`~closure.datasets.DataFrameDataset` in Lightning's
``LightningDataModule`` protocol, absorbing channel selection and
subsampling that previously lived in ``ChannelDataLoader``.
"""

from __future__ import annotations

__all__ = ["ClosureDataModule"]

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import lightning as L

_logger = logging.getLogger("closure.datamodule")

from closure.config import load_paths
from closure.datasets import DataFrameDataset


class ClosureDataModule(L.LightningDataModule):
    """Lightning data module for closure training workflows.

    Parameters
    ----------
    data_folder : str
        Root folder containing the simulation data files.
    norm_folder : str
        Folder to save / load normalisation statistics.
    train_samples_file : str
        CSV file listing training sample filenames.
    val_samples_file : str
        CSV file listing validation sample filenames.
    test_samples_file : str or None
        CSV file listing test sample filenames.
    batch_size : int
        Mini-batch size for all dataloaders.
    num_workers : int
        Number of data-loading workers.
    flatten : bool
        If True, flatten spatial dimensions (pixel-wise MLP mode).
    scaler_features : bool or None
        Enable mean/std normalisation for features.
    scaler_targets : bool or None
        Enable mean/std normalisation for targets.
    prescaler_features : list[str | None] or None
        Per-channel prescaler function names (e.g. ``"log"``).
    prescaler_targets : list[str | None] or None
        Per-channel prescaler function names for targets.
    features_dtype : str
        PyTorch dtype name for features (e.g. ``"float32"``).
    targets_dtype : str
        PyTorch dtype name for targets.
    feature_channel_names : list[str] or None
        Subset of feature channels to use (by name).
    target_channel_names : list[str] or None
        Subset of target channels to use (by name).
    subsample_rate : float
        Controls the effective number of training samples per epoch.
        Values below 1.0 select a random subset (undersampling).
        Values above 1.0 repeat samples so each image is visited
        multiple times per epoch (oversampling); useful with
        ``patch_dim`` to extract many random crops per image.
        Default is 1.0 (use all samples exactly once).
    subsample_seed : int or None
        Seed for reproducible subsampling / oversampling.
    patch_dim : list[int] or None
        ``[width, height]`` for random crop patch extraction.
    read_features_targets_kwargs : dict or None
        Extra keyword arguments forwarded to ``read_pic.read_features_targets``.
    filter_features : dict or None
        Spatial filter configuration for features.
    filter_targets : dict or None
        Spatial filter configuration for targets.
    """

    def __init__(
        self,
        data_folder: str,
        norm_folder: str,
        train_samples_file: str,
        val_samples_file: str,
        test_samples_file: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        flatten: bool = True,
        scaler_features: Optional[bool] = None,
        scaler_targets: Optional[bool] = None,
        prescaler_features: Optional[list[str | None]] = None,
        prescaler_targets: Optional[list[str | None]] = None,
        features_dtype: str = "float32",
        targets_dtype: str = "float32",
        feature_channel_names: Optional[list[str]] = None,
        target_channel_names: Optional[list[str]] = None,
        subsample_rate: float = 1.0,
        subsample_seed: Optional[int] = None,
        patch_dim: Optional[list[int]] = None,
        read_features_targets_kwargs: Optional[dict] = None,
        filter_features: Optional[dict] = None,
        filter_targets: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Will be populated in setup()
        self.train_dataset: DataFrameDataset | None = None
        self.val_dataset: DataFrameDataset | None = None
        self.test_dataset: DataFrameDataset | None = None

        # Channel index caches (populated in setup)
        self.feature_channels: list[int] | None = None
        self.target_channels: list[int] | None = None

    # ------------------------------------------------------------------
    # path resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_path(value: str, paths_yaml_key: str) -> str:
        """Resolve a relative path against the corresponding ``paths.yaml`` root.

        * Absolute paths are returned unchanged.
        * Paths starting with ``./`` or ``../`` are treated as explicitly
          relative to the current working directory (resolved to absolute).
        * All other relative paths (bare identifiers such as
          ``ecsim/Harris/Le``) are joined with the directory indicated by
          *paths_yaml_key* (``"data_dir"`` or ``"work_dir"``) from
          ``paths.yaml``.
        """
        p = Path(value)
        if p.is_absolute():
            return str(p)
        if value.startswith(("./", "../")):
            return str(p.resolve())
        root = Path(load_paths().get(paths_yaml_key, "."))
        return str(root / p)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        hp = self.hparams

        # Resolve relative paths against paths.yaml roots
        data_folder = self._resolve_path(hp.data_folder, "data_dir")
        norm_folder = self._resolve_path(hp.norm_folder, "work_dir")
        train_samples_file = self._resolve_path(hp.train_samples_file, "data_dir")
        val_samples_file = self._resolve_path(hp.val_samples_file, "data_dir")
        test_samples_file = (
            self._resolve_path(hp.test_samples_file, "data_dir")
            if hp.test_samples_file is not None
            else None
        )

        # Build common dataset kwargs
        common = dict(
            data_folder=data_folder,
            norm_folder=norm_folder,
            flatten=hp.flatten,
            features_dtype=hp.features_dtype,
            targets_dtype=hp.targets_dtype,
            scaler_features=hp.scaler_features,
            scaler_targets=hp.scaler_targets,
            prescaler_features=hp.prescaler_features,
            prescaler_targets=hp.prescaler_targets,
            read_features_targets_kwargs=hp.read_features_targets_kwargs,
            filter_features=hp.filter_features,
            filter_targets=hp.filter_targets,
        )

        # Build transform for patch extraction (training only)
        transform = None
        if hp.patch_dim is not None:
            transform = {
                "RandomCrop": {"size": hp.patch_dim},
                "apply": ["train"],
            }

        if stage in ("fit", None):
            t0 = time.perf_counter()
            self.train_dataset = DataFrameDataset(
                samples_file=train_samples_file,
                datalabel="train",
                transform=transform,
                **common,
            )
            self.val_dataset = DataFrameDataset(
                samples_file=val_samples_file,
                datalabel="val",
                **common,
            )
            self._data_load_time_s = time.perf_counter() - t0
            _logger.info(
                "Data loading (train+val) took %.2fs",
                self._data_load_time_s,
            )
            # Resolve channel name → index mappings
            self._resolve_channel_indices(self.train_dataset)

        if stage in ("test", None):
            if test_samples_file is not None:
                self.test_dataset = DataFrameDataset(
                    samples_file=test_samples_file,
                    datalabel="test",
                    **common,
                )
                self._resolve_channel_indices(self.test_dataset)

        if stage == "predict":
            if test_samples_file is not None:
                self.test_dataset = DataFrameDataset(
                    samples_file=test_samples_file,
                    datalabel="test",
                    **common,
                )
                self._resolve_channel_indices(self.test_dataset)

    # ------------------------------------------------------------------
    # dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        dataset = self._maybe_subsample(self.train_dataset)
        return self._make_loader(dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("No test_samples_file configured.")
        return self._make_loader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams

        # Wrap dataset with channel-selection collate if needed
        if self.feature_channels is not None or self.target_channels is not None:
            dataset = _ChannelSubsetDataset(
                dataset, self.feature_channels, self.target_channels
            )

        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            shuffle=shuffle,
            num_workers=hp.num_workers,
            pin_memory=True,
        )

    def _maybe_subsample(self, dataset):
        """Return a ``Subset`` with under- or over-sampling applied.

        When ``subsample_rate < 1.0``, a random subset of the dataset is
        selected (undersampling).  When ``subsample_rate > 1.0``, indices
        are repeated so each sample appears multiple times per epoch
        (oversampling).  This is useful with ``patch_dim`` random cropping
        where each access yields a different random patch.
        """
        hp = self.hparams
        if hp.subsample_rate == 1.0:
            return dataset

        n = len(dataset)
        k = max(1, int(n * hp.subsample_rate))
        rng = np.random.RandomState(hp.subsample_seed)

        if hp.subsample_rate < 1.0:
            indices = rng.choice(n, size=k, replace=False).tolist()
        else:
            # Oversampling: cycle indices so each image is visited
            # subsample_rate times per epoch (matching legacy behaviour).
            indices = (rng.permutation(k) % n).tolist()

        return Subset(dataset, indices)

    def _resolve_channel_indices(self, dataset: DataFrameDataset):
        """Convert channel name lists to integer index lists."""
        hp = self.hparams
        if hp.feature_channel_names is not None and self.feature_channels is None:
            self.feature_channels = [
                dataset.request_features.index(ch) for ch in hp.feature_channel_names
            ]
        if hp.target_channel_names is not None and self.target_channels is None:
            self.target_channels = [
                dataset.request_targets.index(ch) for ch in hp.target_channel_names
            ]


class _ChannelSubsetDataset(torch.utils.data.Dataset):
    """Thin wrapper that selects specific feature/target channels."""

    def __init__(self, dataset, feature_channels, target_channels):
        self.dataset = dataset
        self.feature_channels = feature_channels
        self.target_channels = target_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features, targets = self.dataset[idx]
        if self.feature_channels is not None:
            features = features[self.feature_channels]
        if self.target_channels is not None:
            targets = targets[self.target_channels]
        return features, targets
