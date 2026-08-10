"""Tests for closure.datasets — DataFrameDataset normalization, filtering, prescaling."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from closure.datasets import DataFrameDataset, periodic_binomial_channels


def _build_dataset_with_mock_data(
    tmp_path,
    n_samples=4,
    height=8,
    width=8,
    n_features=2,
    n_targets=1,
    flatten=True,
    prescaler_features=None,
    prescaler_targets=None,
    scaler_features=True,
    scaler_targets=True,
    datalabel="train",
):
    """Build a DataFrameDataset by mocking the file-reading internals."""
    # Create a CSV samples file
    csv_path = os.path.join(tmp_path, f"{datalabel}.csv")
    filenames = [f"sample_{i}.h5" for i in range(n_samples)]
    pd.DataFrame({"filenames": filenames}).to_csv(csv_path, index=False)

    # Synthetic features and targets (N, H, W, C) — before any reshape
    features = np.random.randn(n_samples, height, width, n_features).astype(np.float64)
    targets = np.random.randn(n_samples, height, width, n_targets).astype(np.float64)

    # Patch read_features_targets to return our synthetic data
    with patch("closure.datasets.rp") as mock_rp:
        mock_rp.read_features_targets.return_value = (features, targets)
        mock_rp.build_XY.return_value = (np.linspace(0, 1, width), np.linspace(0, 1, height))

        ds = DataFrameDataset(
            data_folder=str(tmp_path),
            norm_folder=str(tmp_path),
            samples_file=csv_path,
            datalabel=datalabel,
            flatten=flatten,
            prescaler_features=prescaler_features,
            prescaler_targets=prescaler_targets,
            scaler_features=scaler_features,
            scaler_targets=scaler_targets,
            read_features_targets_kwargs={
                "request_features": [f"f{i}" for i in range(n_features)],
                "request_targets": [f"t{i}" for i in range(n_targets)],
            },
        )
    return ds, features, targets


class TestDataFrameDatasetBasic:
    def test_len(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_samples=6, flatten=True)
        # flatten: n_samples * H * W
        assert len(ds) == 6 * 8 * 8

    def test_len_nchw(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_samples=4, flatten=False)
        assert len(ds) == 4

    def test_getitem_returns_tensors(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, flatten=True)
        features, targets = ds[0]
        assert isinstance(features, torch.Tensor)
        assert isinstance(targets, torch.Tensor)

    def test_feature_shape_flatten(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_features=3, flatten=True)
        f, t = ds[0]
        assert f.shape == (3,)
        assert t.shape == (1,)

    def test_feature_shape_nchw(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_features=3, flatten=False)
        f, t = ds[0]
        assert f.shape == (3, 8, 8)
        assert t.shape == (1, 8, 8)


class TestPeriodicBinomialChannels:
    def test_matches_four_explicit_periodic_passes_and_only_selected_channel(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(2, 9, 11, 3)).astype(np.float32)
        original = data.copy()
        expected = data.copy()
        for _ in range(4):
            selected = expected[..., 1]
            expected[..., 1] = (
                4.0 * selected
                + 2.0 * (
                    np.roll(selected, 1, axis=1)
                    + np.roll(selected, -1, axis=1)
                    + np.roll(selected, 1, axis=2)
                    + np.roll(selected, -1, axis=2)
                )
                + np.roll(np.roll(selected, 1, axis=1), 1, axis=2)
                + np.roll(np.roll(selected, 1, axis=1), -1, axis=2)
                + np.roll(np.roll(selected, -1, axis=1), 1, axis=2)
                + np.roll(np.roll(selected, -1, axis=1), -1, axis=2)
            ) / 16.0

        actual = periodic_binomial_channels(
            data, channel_indices=[1], passes=4
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(actual[..., 0], data[..., 0])
        np.testing.assert_array_equal(actual[..., 2], data[..., 2])
        np.testing.assert_array_equal(data, original)

    def test_named_dataset_filter_rejects_unknown_channel(self, tmp_path):
        csv_path = tmp_path / "train.csv"
        pd.DataFrame({"filenames": ["sample.h5"]}).to_csv(csv_path, index=False)
        with patch("closure.datasets.rp") as mock_rp:
            mock_rp.read_features_targets.return_value = (
                np.zeros((1, 4, 4, 2)), np.zeros((1, 4, 4, 1))
            )
            with pytest.raises(ValueError, match="unknown=.*Wxx_e"):
                DataFrameDataset(
                    data_folder=str(tmp_path), norm_folder=str(tmp_path),
                    samples_file=str(csv_path), scaler_features=False,
                    scaler_targets=False,
                    read_features_targets_kwargs={
                        "request_features": ["rho_e", "Bx"],
                        "request_targets": ["Pxx_e"],
                    },
                    filter_features={
                        "name": "periodic_binomial_channels",
                        "channels": ["Wxx_e"], "passes": 4,
                    },
                )


class TestNormalization:
    def test_features_normalized_mean_std(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_samples=10, flatten=True, scaler_features=True)
        # After normalization, per-channel mean ≈ 0, std ≈ 1
        feats = ds.features.numpy()
        for c in range(feats.shape[-1]):
            assert abs(feats[:, c].mean()) < 0.2  # approximate
            assert abs(feats[:, c].std() - 1.0) < 0.2

    def test_targets_normalized(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_samples=10, flatten=True, scaler_targets=True)
        tgts = ds.targets.numpy()
        for c in range(tgts.shape[-1]):
            assert abs(tgts[:, c].mean()) < 0.2
            assert abs(tgts[:, c].std() - 1.0) < 0.2

    def test_no_normalization(self, tmp_path):
        ds, raw_f, _ = _build_dataset_with_mock_data(
            tmp_path, n_samples=4, flatten=True, scaler_features=False, scaler_targets=False
        )
        # Without normalization, data should be close to raw (just reshaped)
        assert ds.features_mean is None or ds.features_mean is False


class TestPrescaling:
    def test_log_prescaler(self, tmp_path):
        """Verify that log prescaler is applied without error."""
        ds, _, _ = _build_dataset_with_mock_data(
            tmp_path,
            n_samples=4,
            flatten=True,
            prescaler_features=["log", None],
            scaler_features=False,
            scaler_targets=False,
        )
        # If log was applied to channel 0, features should exist (no NaN crash for negative vals
        # is a known limitation — but the dataset should still be created)
        assert ds.features is not None


class TestAttributes:
    def test_request_features(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_features=3)
        assert ds.request_features == ["f0", "f1", "f2"]

    def test_request_targets(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_targets=2)
        assert ds.request_targets == ["t0", "t1"]

    def test_prescaler_targets_stored(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, prescaler_targets=[None])
        assert ds.prescaler_targets is not None

    def test_flatten_attribute(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, flatten=True)
        assert ds.flatten is True

    def test_features_shape_stored(self, tmp_path):
        ds, _, _ = _build_dataset_with_mock_data(tmp_path, n_samples=4, height=8, width=8, n_features=2)
        assert ds.features_shape is not None
