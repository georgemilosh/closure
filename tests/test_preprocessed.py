"""Tests for PreprocessedChunkDataset and ChunkOrderedSampler.

Covers:
- Preprocessing creates valid metadata + chunk files.
- __getitem__ returns tensors with correct shapes (flatten=False and flatten=True).
- Pre-normalized values match LazyNPZDataFrameDataset (parity test).
- Stale cache is reused (no double preprocessing).
- ChunkOrderedSampler covers every index exactly once per epoch.
- ChunkOrderedSampler set_epoch changes the iteration order.
- End-to-end CLI fit with loading_mode=preprocessed (FCNN + MLP).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml

from closure.datasets import (
    ChunkOrderedSampler,
    LazyNPZDataFrameDataset,
    PreprocessedChunkDataset,
    _preprocessing_fingerprint,
)

# Shared NPZ field plumbing (same fields as test_cli_fit_lazy.py).
_RFT_KWARGS = {
    "fields_to_read": {"B": True, "E": True},
    "request_features": ["Bx", "By", "Bz"],
    "request_targets": ["Ex", "Ey", "Ez"],
    "verbose": False,
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_preprocessed(
    tiny_npz_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    datalabel: str = "train",
    flatten: bool = False,
) -> PreprocessedChunkDataset:
    norm_dir = tmp_path / "norm"
    norm_dir.mkdir(exist_ok=True)
    ssd_dir = tmp_path / "ssd"
    ssd_dir.mkdir(exist_ok=True)
    csv = tiny_npz_dir / f"{datalabel}.csv"
    return PreprocessedChunkDataset(
        data_folder=str(tiny_npz_dir),
        norm_folder=str(norm_dir),
        samples_file=str(csv),
        ssd_cache_dir=str(ssd_dir),
        features_dtype="float32",
        targets_dtype="float32",
        scaler_features=True,
        scaler_targets=True,
        datalabel=datalabel,
        flatten=flatten,
        read_features_targets_kwargs=_RFT_KWARGS,
        chunk_cache_size=2,
        preprocess_chunk_size_gb=0.001,  # tiny budget → many small chunks
    )


# -----------------------------------------------------------------------
# PreprocessedChunkDataset unit tests
# -----------------------------------------------------------------------

class TestPreprocessedChunkDataset:
    def test_fingerprint_covers_all_cached_preprocessing(self, tmp_path):
        common = dict(
            samples_file="train.csv",
            request_features=["Bx", "By", "Bz"],
            request_targets=["Ex", "Ey", "Ez"],
            prescaler_features=[None, None, None],
            prescaler_targets=[None, None, None],
            alfven_units=True,
            read_features_targets_kwargs=_RFT_KWARGS,
            scaler_features=False,
            scaler_targets=False,
            norm_folder=str(tmp_path / "norm"),
            features_dtype_numpy=np.float32,
            targets_dtype_numpy=np.float32,
        )
        baseline = _preprocessing_fingerprint(**common)
        filtered = _preprocessing_fingerprint(
            **common,
            filter_features={
                "name": "periodic_binomial_channels",
                "channels": ["Bx"],
                "passes": 4,
                "axes": [1, 2],
            },
        )
        normalized = _preprocessing_fingerprint(
            **{**common, "scaler_features": True}
        )
        different_reader = _preprocessing_fingerprint(
            **{
                **common,
                "read_features_targets_kwargs": {
                    **_RFT_KWARGS,
                    "choose_x": [0, 4],
                },
            }
        )
        assert len({baseline, filtered, normalized, different_reader}) == 4

    def test_metadata_created(self, tiny_npz_dir, tmp_path):
        ds = _make_preprocessed(tiny_npz_dir, tmp_path)
        assert ds._num_chunks >= 1
        assert ds.num_files == 5  # tiny_npz_dir fixture has 5 train files
        assert (pathlib.Path(ds._meta_path)).exists()

    def test_chunk_files_created(self, tiny_npz_dir, tmp_path):
        ds = _make_preprocessed(tiny_npz_dir, tmp_path)
        for i in range(ds._num_chunks):
            p = pathlib.Path(ds._chunk_dir) / f"chunk_{i:04d}.pt"
            assert p.exists(), f"chunk_{i:04d}.pt missing"

    def test_norm_pkl_created(self, tiny_npz_dir, tmp_path):
        norm_dir = tmp_path / "norm"
        norm_dir.mkdir(exist_ok=True)
        ssd_dir = tmp_path / "ssd"
        ssd_dir.mkdir(exist_ok=True)
        PreprocessedChunkDataset(
            data_folder=str(tiny_npz_dir),
            norm_folder=str(norm_dir),
            samples_file=str(tiny_npz_dir / "train.csv"),
            ssd_cache_dir=str(ssd_dir),
            scaler_features=True,
            scaler_targets=True,
            datalabel="train",
            flatten=False,
            read_features_targets_kwargs=_RFT_KWARGS,
            chunk_cache_size=1,
        )
        assert (norm_dir / "X.pkl").exists()
        assert (norm_dir / "y.pkl").exists()

    def test_len_flatten_false(self, tiny_npz_dir, tmp_path):
        ds = _make_preprocessed(tiny_npz_dir, tmp_path, flatten=False)
        assert len(ds) == ds.num_files  # 5 train files

    def test_len_flatten_true(self, tiny_npz_dir, tmp_path):
        ds = _make_preprocessed(tiny_npz_dir, tmp_path, flatten=True)
        # After rp off-by-one slice (8→7 each axis) _pixels_per_file = _h * _w
        assert len(ds) == ds.num_files * ds._pixels_per_file

    def test_getitem_shapes_flatten_false(self, tiny_npz_dir, tmp_path):
        ds = _make_preprocessed(tiny_npz_dir, tmp_path, flatten=False)
        feat, targ = ds[0]
        assert feat.shape == (3, 7, 7)  # (C_f, H-1, W-1) after rp off-by-one
        assert targ.shape == (3, 7, 7)
        assert feat.dtype == torch.float32

    def test_getitem_shapes_flatten_true(self, tiny_npz_dir, tmp_path):
        ds = _make_preprocessed(tiny_npz_dir, tmp_path, flatten=True)
        feat, targ = ds[0]
        assert feat.ndim == 1 and feat.shape[0] == 3
        assert targ.ndim == 1 and targ.shape[0] == 3

    def test_cache_reuse(self, tiny_npz_dir, tmp_path):
        """Second construction skips preprocessing (metadata already exists)."""
        ds1 = _make_preprocessed(tiny_npz_dir, tmp_path)
        mtime1 = pathlib.Path(ds1._meta_path).stat().st_mtime

        ds2 = _make_preprocessed(tiny_npz_dir, tmp_path)
        mtime2 = pathlib.Path(ds2._meta_path).stat().st_mtime
        assert mtime1 == mtime2, "metadata was rewritten on second construction"

    def test_parity_with_lazy_flatten_false(self, tiny_npz_dir, tmp_path):
        """Pre-normalized values must match LazyNPZDataFrameDataset."""
        norm_dir = tmp_path / "norm"
        norm_dir.mkdir(exist_ok=True)
        ssd_dir = tmp_path / "ssd"
        ssd_dir.mkdir(exist_ok=True)

        pre = PreprocessedChunkDataset(
            data_folder=str(tiny_npz_dir),
            norm_folder=str(norm_dir),
            samples_file=str(tiny_npz_dir / "train.csv"),
            ssd_cache_dir=str(ssd_dir),
            scaler_features=True,
            scaler_targets=True,
            datalabel="train",
            flatten=False,
            read_features_targets_kwargs=_RFT_KWARGS,
            chunk_cache_size=2,
        )
        lazy = LazyNPZDataFrameDataset(
            data_folder=str(tiny_npz_dir),
            norm_folder=str(norm_dir),
            samples_file=str(tiny_npz_dir / "train.csv"),
            scaler_features=True,
            scaler_targets=True,
            datalabel="train",
            flatten=False,
            read_features_targets_kwargs=_RFT_KWARGS,
            sample_cache_size=1,
        )
        # pre[slot] stores CSV file pre._file_perm[slot]; compare against lazy[csv_idx].
        for slot in range(pre.num_files):
            csv_idx = pre._file_perm[slot]
            pf, pt = pre[slot]
            lf, lt = lazy[csv_idx]
            np.testing.assert_allclose(pf.numpy(), lf.numpy(), atol=1e-5,
                                       err_msg=f"feature mismatch at slot {slot} (csv {csv_idx})")
            np.testing.assert_allclose(pt.numpy(), lt.numpy(), atol=1e-5,
                                       err_msg=f"target mismatch at slot {slot} (csv {csv_idx})")

    def test_val_dataset_uses_train_norms(self, tiny_npz_dir, tmp_path):
        """Val preprocessing must reuse norm stats computed on train."""
        norm_dir = tmp_path / "norm"
        norm_dir.mkdir(exist_ok=True)
        ssd_dir = tmp_path / "ssd"
        ssd_dir.mkdir(exist_ok=True)
        common = dict(
            data_folder=str(tiny_npz_dir),
            norm_folder=str(norm_dir),
            ssd_cache_dir=str(ssd_dir),
            scaler_features=True,
            scaler_targets=True,
            flatten=False,
            read_features_targets_kwargs=_RFT_KWARGS,
            chunk_cache_size=1,
        )
        PreprocessedChunkDataset(samples_file=str(tiny_npz_dir / "train.csv"),
                                 datalabel="train", **common)
        # Val should succeed because train already wrote X.pkl/y.pkl.
        val_ds = PreprocessedChunkDataset(samples_file=str(tiny_npz_dir / "val.csv"),
                                          datalabel="val", **common)
        assert val_ds.num_files == 2  # tiny_npz fixture has 2 val files


# -----------------------------------------------------------------------
# ChunkOrderedSampler unit tests
# -----------------------------------------------------------------------

class TestChunkOrderedSampler:
    def _sampler(self, chunk_sizes, ppf=1, oversample=1, shuffle=True, seed=0):
        return ChunkOrderedSampler(
            chunk_sizes=chunk_sizes,
            pixels_per_file=ppf,
            oversample=oversample,
            shuffle=shuffle,
            seed=seed,
        )

    def test_len_file_mode(self):
        s = self._sampler([3, 4, 2], ppf=1, oversample=1)
        assert len(s) == 9

    def test_len_pixel_mode(self):
        s = self._sampler([3, 4], ppf=64, oversample=1)
        assert len(s) == 7 * 64

    def test_len_oversample(self):
        s = self._sampler([3, 2], ppf=1, oversample=5)
        assert len(s) == 25

    def test_covers_all_indices_file_mode(self):
        s = self._sampler([3, 4, 2], ppf=1, oversample=1)
        indices = list(iter(s))
        assert sorted(indices) == list(range(9))

    def test_covers_all_indices_pixel_mode(self):
        ppf = 4
        s = self._sampler([2, 3], ppf=ppf, oversample=1)
        indices = list(iter(s))
        assert sorted(indices) == list(range(5 * ppf))

    def test_covers_all_indices_oversample(self):
        oversample = 3
        s = self._sampler([2, 2], ppf=1, oversample=oversample)
        indices = list(iter(s))
        assert len(indices) == 4 * oversample
        # Each index should appear exactly oversample times.
        from collections import Counter
        cnt = Counter(indices)
        assert all(v == oversample for v in cnt.values())

    def test_set_epoch_changes_order(self):
        s = self._sampler([5, 5, 5], ppf=1, oversample=1, shuffle=True, seed=42)
        s.set_epoch(0)
        order0 = list(iter(s))
        s.set_epoch(1)
        order1 = list(iter(s))
        assert order0 != order1, "different epochs should produce different orders"

    def test_no_shuffle_deterministic(self):
        s = self._sampler([3, 2], ppf=1, oversample=1, shuffle=False)
        assert list(iter(s)) == list(iter(s))

    def test_chunk_contiguity(self):
        """All indices from chunk 0 come before all indices from chunk 1 (no-shuffle)."""
        s = self._sampler([3, 4], ppf=1, oversample=1, shuffle=False)
        indices = list(iter(s))
        # chunk 0: files 0,1,2; chunk 1: files 3,4,5,6
        chunk0_done = False
        for idx in indices:
            if idx >= 3:  # entered chunk 1
                chunk0_done = True
            if chunk0_done and idx < 3:
                pytest.fail("chunk 0 index appeared after chunk 1 index")


# -----------------------------------------------------------------------
# End-to-end CLI tests
# -----------------------------------------------------------------------

def _base_cfg(tiny_npz_dir: pathlib.Path, output_dir: pathlib.Path,
              ssd_dir: pathlib.Path) -> dict:
    return {
        "seed_everything": 42,
        "model": {
            "criterion": "MSELoss",
            "optimizer": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0,
            "scheduler": None,
        },
        "data": {
            "data_folder": str(tiny_npz_dir),
            "norm_folder": str(output_dir),
            "train_samples_file": str(tiny_npz_dir / "train.csv"),
            "val_samples_file": str(tiny_npz_dir / "val.csv"),
            "batch_size": 2,
            "num_workers": 0,
            "scaler_features": True,
            "scaler_targets": True,
            "features_dtype": "float32",
            "targets_dtype": "float32",
            "loading_mode": "preprocessed",
            "ssd_cache_dir": str(ssd_dir),
            "chunk_cache_size": 1,
            "read_features_targets_kwargs": _RFT_KWARGS,
        },
        "trainer": {
            "fast_dev_run": True,
            "accelerator": "cpu",
            "devices": 1,
            "default_root_dir": str(output_dir),
            "enable_progress_bar": False,
            "logger": False,
            "enable_checkpointing": False,
        },
    }


def _write_yaml(tmp_path: pathlib.Path, cfg: dict) -> pathlib.Path:
    p = tmp_path / "fit_preprocessed.yaml"
    p.write_text(yaml.dump(cfg, default_flow_style=False))
    return p


def _run_cli(config_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["closure-train", "fit", f"--config={config_path}"]
    )
    from closure.cli import main
    try:
        main()
    except SystemExit as exc:
        assert exc.code in (None, 0), f"CLI exited with code {exc.code}"


class TestCLIFitPreprocessed:
    def test_fit_flatten_false_fcnn(self, tiny_npz_dir, tmp_path, monkeypatch):
        output_dir = tmp_path / "out_fcnn"
        output_dir.mkdir()
        ssd_dir = tmp_path / "ssd"
        ssd_dir.mkdir()
        cfg = _base_cfg(tiny_npz_dir, output_dir, ssd_dir)
        cfg["model"]["network"] = {
            "class_path": "closure.models.FCNN",
            "init_args": {
                "channels": [3, 8, 3],
                "kernels": [3, 3],
                "activations": ["ReLU", None],
            },
        }
        cfg["data"]["flatten"] = False
        cfg["data"]["patch_dim"] = [4, 4]
        cfg["data"]["batch_size"] = 2
        _run_cli(_write_yaml(tmp_path, cfg), monkeypatch)
        assert (output_dir / "X.pkl").exists()
        assert (output_dir / "y.pkl").exists()

    def test_fit_flatten_true_mlp(self, tiny_npz_dir, tmp_path, monkeypatch):
        output_dir = tmp_path / "out_mlp"
        output_dir.mkdir()
        ssd_dir = tmp_path / "ssd"
        ssd_dir.mkdir()
        cfg = _base_cfg(tiny_npz_dir, output_dir, ssd_dir)
        cfg["model"]["network"] = {
            "class_path": "closure.models.MLP",
            "init_args": {
                "feature_dims": [3, 8, 3],
                "activations": ["ReLU", None],
            },
        }
        cfg["data"]["flatten"] = True
        cfg["data"]["batch_size"] = 8
        _run_cli(_write_yaml(tmp_path, cfg), monkeypatch)
        assert (output_dir / "X.pkl").exists()
        assert (output_dir / "y.pkl").exists()
