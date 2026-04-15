"""Tests verifying that the refactored Lightning stack can replicate legacy configs.

The attached legacy config (Harris/Le/Le2GEM15ppc/default/P/4lrs_es500/config.json)
exercises:

- FCNN with ``flatten=False``, ``patch_dim=[32, 32]``, ``subsample_rate=160``
- the oversampling + random-crop workflow where each image is visited 160×/epoch
- model architecture: ``channels=[10, 128, 64, 32, 6]``, ``kernels=[3, 5, 5, 3]``
- ReduceLROnPlateau scheduler with the same kwargs

These tests ensure the refactored ``ClosureDataModule`` and ``ClosureLitModule``
accept an equivalent configuration and produce the correct dataloader sizes,
model shapes, and scheduling behaviour.
"""

from __future__ import annotations

import os
from collections import Counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from closure.datamodule import ClosureDataModule
from closure.models import FCNN
from closure.module import ClosureLitModule


# -------------------------------------------------------------------
# Legacy-equivalent configuration values (from 4lrs_es500/config.json)
# -------------------------------------------------------------------
N_FEATURES = 10
N_TARGETS = 6
LEGACY_CHANNELS = [10, 128, 64, 32, 6]
LEGACY_KERNELS = [3, 5, 5, 3]
LEGACY_ACTIVATIONS = ["ReLU", "ReLU", "ReLU", None]
LEGACY_BATCH_NORMS = [True, True, True, False]
LEGACY_BATCH_SIZE = 32
LEGACY_PATCH_DIM = [32, 32]
LEGACY_SUBSAMPLE_RATE = 160
LEGACY_SUBSAMPLE_SEED = 42
LEGACY_PRESCALER_TARGETS = ["log", "log", "log", "arcsinh", "arcsinh", "arcsinh"]


# -------------------------------------------------------------------
# Helper: build a ClosureDataModule with mocked data that mirrors the
# legacy NCHW (non-flattened, 2-D field) layout.
# -------------------------------------------------------------------
def _build_nchw_datamodule(
    tmp_path,
    n_images=5,
    height=64,
    width=64,
    subsample_rate=LEGACY_SUBSAMPLE_RATE,
    subsample_seed=LEGACY_SUBSAMPLE_SEED,
    patch_dim=None,
    batch_size=LEGACY_BATCH_SIZE,
):
    """Create a ClosureDataModule backed by synthetic NCHW data.

    ``read_features_targets`` is mocked so no real HDF5 files are needed.
    """
    # Write sample CSVs
    filenames = [f"sample_{i}.h5" for i in range(n_images)]
    for split in ("train", "val", "test"):
        pd.DataFrame({"filenames": filenames}).to_csv(
            os.path.join(tmp_path, f"{split}.csv"), index=False
        )

    # Synthetic data: (N, H, W, C) before reshape
    features = np.abs(np.random.RandomState(0).randn(
        n_images, height, width, N_FEATURES
    )).astype(np.float64) + 1.0
    targets = np.abs(np.random.RandomState(1).randn(
        n_images, height, width, N_TARGETS
    )).astype(np.float64) + 1.0  # positive so log prescaler works

    patch_dim_arg = patch_dim if patch_dim is not None else LEGACY_PATCH_DIM

    dm = ClosureDataModule(
        data_folder=str(tmp_path),
        norm_folder=str(tmp_path),
        train_samples_file=os.path.join(tmp_path, "train.csv"),
        val_samples_file=os.path.join(tmp_path, "val.csv"),
        test_samples_file=os.path.join(tmp_path, "test.csv"),
        batch_size=batch_size,
        num_workers=0,
        flatten=False,
        scaler_features=True,
        scaler_targets=True,
        prescaler_targets=LEGACY_PRESCALER_TARGETS,
        subsample_rate=subsample_rate,
        subsample_seed=subsample_seed,
        patch_dim=patch_dim_arg,
        read_features_targets_kwargs={
            "request_features": [f"f{i}" for i in range(N_FEATURES)],
            "request_targets": [f"t{i}" for i in range(N_TARGETS)],
        },
    )

    # Patch the file-reading layer so setup() works without real HDF5 data
    with patch("closure.datasets.rp") as mock_rp:
        mock_rp.read_features_targets.return_value = (features, targets)
        mock_rp.build_XY.return_value = (
            np.linspace(0, 1, width),
            np.linspace(0, 1, height),
        )
        dm.setup("fit")
        dm.setup("test")

    return dm


# ===================================================================
# Test: oversampling produces the correct number of samples
# ===================================================================
class TestOversampling:
    def test_oversample_length(self, tmp_path):
        """subsample_rate=160 on 5 images → 800 indices in the train loader."""
        dm = _build_nchw_datamodule(tmp_path, n_images=5, subsample_rate=160)
        train_ds = dm._maybe_subsample(dm.train_dataset)
        assert len(train_ds) == 5 * 160

    def test_oversample_uniform_coverage(self, tmp_path):
        """Every image should be visited exactly subsample_rate times."""
        n_images = 5
        dm = _build_nchw_datamodule(tmp_path, n_images=n_images, subsample_rate=160)
        train_ds = dm._maybe_subsample(dm.train_dataset)
        counts = Counter(train_ds.indices)
        assert set(counts.keys()) == set(range(n_images))
        assert all(c == 160 for c in counts.values())

    def test_oversample_different_from_identity(self, tmp_path):
        """Oversampled dataset is a Subset, not the raw dataset."""
        dm = _build_nchw_datamodule(tmp_path, n_images=5, subsample_rate=160)
        train_ds = dm._maybe_subsample(dm.train_dataset)
        assert isinstance(train_ds, torch.utils.data.Subset)

    def test_subsample_rate_1_returns_raw_dataset(self, tmp_path):
        """subsample_rate=1.0 must return the original dataset unchanged."""
        dm = _build_nchw_datamodule(tmp_path, n_images=5, subsample_rate=1.0)
        train_ds = dm._maybe_subsample(dm.train_dataset)
        assert train_ds is dm.train_dataset

    def test_undersample(self, tmp_path):
        """subsample_rate=0.4 on 5 images → 2 unique indices."""
        dm = _build_nchw_datamodule(tmp_path, n_images=5, subsample_rate=0.4)
        train_ds = dm._maybe_subsample(dm.train_dataset)
        assert len(train_ds) == 2
        assert len(set(train_ds.indices)) == 2  # no duplicates

    def test_reproducibility(self, tmp_path):
        """Same seed → same oversampled indices."""
        dm1 = _build_nchw_datamodule(tmp_path, n_images=5, subsample_rate=160, subsample_seed=42)
        dm2 = _build_nchw_datamodule(tmp_path, n_images=5, subsample_rate=160, subsample_seed=42)
        idx1 = dm1._maybe_subsample(dm1.train_dataset).indices
        idx2 = dm2._maybe_subsample(dm2.train_dataset).indices
        assert idx1 == idx2


# ===================================================================
# Test: FCNN model matches the legacy 4lrs architecture
# ===================================================================
class TestLegacyFCNNArchitecture:
    def test_forward_shape_with_legacy_config(self):
        """FCNN(10, 128, 64, 32, 6) with patch_dim=[32, 32] → (B, 6, 32, 32)."""
        model = FCNN(
            channels=LEGACY_CHANNELS,
            kernels=LEGACY_KERNELS,
            activations=LEGACY_ACTIVATIONS,
            batch_norms=LEGACY_BATCH_NORMS,
        )
        x = torch.randn(LEGACY_BATCH_SIZE, N_FEATURES, *LEGACY_PATCH_DIM)
        out = model(x)
        assert out.shape == (LEGACY_BATCH_SIZE, N_TARGETS, *LEGACY_PATCH_DIM)

    def test_batchnorm_layers_present(self):
        model = FCNN(
            channels=LEGACY_CHANNELS,
            kernels=LEGACY_KERNELS,
            activations=LEGACY_ACTIVATIONS,
            batch_norms=LEGACY_BATCH_NORMS,
        )
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        # batch_norms=[True, True, True, False] → 3 BN layers
        assert len(bn_layers) == 3


# ===================================================================
# Test: ClosureLitModule with legacy-equivalent settings
# ===================================================================
class TestLegacyLitModule:
    def _build_module(self):
        network = FCNN(
            channels=LEGACY_CHANNELS,
            kernels=LEGACY_KERNELS,
            activations=LEGACY_ACTIVATIONS,
            batch_norms=LEGACY_BATCH_NORMS,
        )
        return ClosureLitModule(
            network=network,
            criterion="MSELoss",
            optimizer="Adam",
            lr=0.001,
            weight_decay=1e-5,
            scheduler="ReduceLROnPlateau",
            scheduler_kwargs={
                "mode": "min",
                "factor": 0.2,
                "patience": 25,
                "min_lr": 1e-5,
            },
        )

    def test_training_step_on_patch(self):
        module = self._build_module()
        x = torch.randn(LEGACY_BATCH_SIZE, N_FEATURES, *LEGACY_PATCH_DIM)
        y = torch.randn(LEGACY_BATCH_SIZE, N_TARGETS, *LEGACY_PATCH_DIM)
        loss = module.training_step((x, y), 0)
        assert loss.ndim == 0  # scalar

    def test_scheduler_config(self):
        module = self._build_module()
        cfg = module.configure_optimizers()
        assert "lr_scheduler" in cfg
        sched = cfg["lr_scheduler"]["scheduler"]
        assert sched.__class__.__name__ == "ReduceLROnPlateau"
        assert sched.patience == 25
        assert sched.factor == pytest.approx(0.2)


# ===================================================================
# Test: train_dataloader batch shape matches legacy expectations
# ===================================================================
class TestLegacyTrainDataloader:
    def test_batch_shape(self, tmp_path):
        """Train dataloader must yield patches of shape (B, C, pH, pW)."""
        dm = _build_nchw_datamodule(
            tmp_path,
            n_images=3,
            height=64,
            width=64,
            subsample_rate=10,  # small for fast iteration
            batch_size=4,
            patch_dim=[32, 32],
        )
        loader = dm.train_dataloader()
        features, targets = next(iter(loader))
        assert features.shape == (4, N_FEATURES, 32, 32)
        assert targets.shape == (4, N_TARGETS, 32, 32)

    def test_total_batches(self, tmp_path):
        """Oversampled loader should have ceil(n_images * rate / batch_size) batches."""
        n_images, rate, bs = 5, 20, 10
        dm = _build_nchw_datamodule(
            tmp_path,
            n_images=n_images,
            subsample_rate=rate,
            batch_size=bs,
            patch_dim=[32, 32],
        )
        loader = dm.train_dataloader()
        n_batches = sum(1 for _ in loader)
        expected = n_images * rate // bs  # 100 / 10 = 10
        assert n_batches == expected

    def test_val_loader_no_oversampling(self, tmp_path):
        """Validation loader must NOT oversample (uses raw dataset)."""
        dm = _build_nchw_datamodule(
            tmp_path,
            n_images=5,
            subsample_rate=160,
        )
        val_loader = dm.val_dataloader()
        # val uses raw dataset → 5 images
        n_val_samples = len(val_loader.dataset)
        assert n_val_samples == 5
