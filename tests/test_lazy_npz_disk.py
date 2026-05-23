"""Integration tests for LazyNPZDataFrameDataset against real ``.npz`` files.

Complements ``test_lazy_npz.py`` (which mocks ``rp.read_features_targets``)
by exercising the actual disk decode path through
``closure.read_pic.read_features_targets`` / ``read_data`` / ``read_fieldname``.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pytest
import torch

from closure.datasets import DataFrameDataset, LazyNPZDataFrameDataset


# Shared kwargs read from the tiny on-disk NPZ fixture (see conftest).
RFT_KWARGS = {
    "fields_to_read": {"B": True, "E": True},
    "request_features": ["Bx", "By", "Bz"],
    "request_targets": ["Ex", "Ey", "Ez"],
    "verbose": False,
}


def _build(cls, data_folder, samples_file, norm_folder, flatten, **extra):
    norm_folder.mkdir(parents=True, exist_ok=True)
    return cls(
        data_folder=str(data_folder),
        norm_folder=str(norm_folder),
        samples_file=str(samples_file),
        datalabel="train",
        flatten=flatten,
        scaler_features=True,
        scaler_targets=True,
        read_features_targets_kwargs=RFT_KWARGS,
        **extra,
    )


# ----------------------------------------------------------------------
# Eager vs lazy parity on real .npz files
# ----------------------------------------------------------------------
class TestEagerLazyDiskParity:
    def test_normalization_stats_match(self, tiny_npz_dir):
        eager_norm = tiny_npz_dir / "norm_eager"
        lazy_norm = tiny_npz_dir / "norm_lazy"
        _build(
            DataFrameDataset,
            tiny_npz_dir,
            tiny_npz_dir / "train.csv",
            eager_norm,
            flatten=True,
        )
        _build(
            LazyNPZDataFrameDataset,
            tiny_npz_dir,
            tiny_npz_dir / "train.csv",
            lazy_norm,
            flatten=True,
            sample_cache_size=2,
        )

        for fname in ("X.pkl", "y.pkl"):
            e_mean, e_std = joblib.load(eager_norm / fname)
            l_mean, l_std = joblib.load(lazy_norm / fname)
            np.testing.assert_allclose(e_mean, l_mean, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(e_std, l_std, rtol=1e-5, atol=1e-6)

    def test_getitem_parity_flatten_true(self, tiny_npz_dir):
        eager = _build(
            DataFrameDataset,
            tiny_npz_dir,
            tiny_npz_dir / "train.csv",
            tiny_npz_dir / "norm_eager",
            flatten=True,
        )
        lazy = _build(
            LazyNPZDataFrameDataset,
            tiny_npz_dir,
            tiny_npz_dir / "train.csv",
            tiny_npz_dir / "norm_lazy",
            flatten=True,
            sample_cache_size=2,
        )
        assert len(eager) == len(lazy)
        # Spot-check across file boundaries.
        pixels_per_file = 8 * 8
        for idx in [0, 1, pixels_per_file - 1, pixels_per_file, 3 * pixels_per_file + 7, len(eager) - 1]:
            fe, te = eager[idx]
            fl, tl = lazy[idx]
            torch.testing.assert_close(fe, fl, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(te, tl, rtol=1e-5, atol=1e-6)

    def test_getitem_parity_flatten_false(self, tiny_npz_dir):
        eager = _build(
            DataFrameDataset,
            tiny_npz_dir,
            tiny_npz_dir / "train.csv",
            tiny_npz_dir / "norm_eager",
            flatten=False,
        )
        lazy = _build(
            LazyNPZDataFrameDataset,
            tiny_npz_dir,
            tiny_npz_dir / "train.csv",
            tiny_npz_dir / "norm_lazy",
            flatten=False,
            sample_cache_size=2,
        )
        assert len(eager) == len(lazy)
        for idx in range(len(eager)):
            fe, te = eager[idx]
            fl, tl = lazy[idx]
            assert fl.shape == fe.shape
            assert tl.shape == te.shape
            torch.testing.assert_close(fe, fl, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(te, tl, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# LRU cache actually avoids repeat disk reads
# ----------------------------------------------------------------------
class TestDiskLRUCache:
    def test_cache_hit_avoids_second_load(self, tiny_npz_dir, monkeypatch):
        lazy = _build(
            LazyNPZDataFrameDataset,
            tiny_npz_dir,
            tiny_npz_dir / "train.csv",
            tiny_npz_dir / "norm_lazy",
            flatten=False,
            sample_cache_size=3,
        )

        # Wrap np.load to count actual decodes after dataset is constructed
        # (so probing/normalization reads don't pollute the count).
        import numpy as _np

        original = _np.load
        calls = {"n": 0}

        def counting_load(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(_np, "load", counting_load)

        # First read of file 0 must hit disk.
        _ = lazy[0]
        first = calls["n"]
        assert first > 0, "expected at least one np.load call for first access"

        # Repeated reads of file 0 should not trigger any new np.load.
        _ = lazy[0]
        _ = lazy[0]
        assert calls["n"] == first, (
            f"cache miss: expected {first} np.load calls, got {calls['n']}"
        )

        # Reading a different file must trigger more np.load calls.
        _ = lazy[2]
        assert calls["n"] > first

    def test_cache_capacity_one_evicts(self, tiny_npz_dir, monkeypatch):
        lazy = _build(
            LazyNPZDataFrameDataset,
            tiny_npz_dir,
            tiny_npz_dir / "train.csv",
            tiny_npz_dir / "norm_lazy",
            flatten=False,
            sample_cache_size=1,
        )

        import numpy as _np

        original = _np.load
        calls = {"n": 0}

        def counting_load(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(_np, "load", counting_load)

        _ = lazy[0]
        n0 = calls["n"]
        _ = lazy[1]
        n1 = calls["n"]
        assert n1 > n0  # second file forced a load
        # Going back to file 0 evicts file 1's cached version and reloads.
        _ = lazy[0]
        n2 = calls["n"]
        assert n2 > n1, "cache capacity=1 should evict and reload"
