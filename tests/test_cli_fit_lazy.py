"""End-to-end CLI tests for ``closure-train fit`` with ``loading_mode: lazy_npz``.

Exercises the full Lightning CLI on top of the on-disk ``tiny_npz_dir``
fixture, hitting both lazy code paths:

* ``flatten=False`` -> :class:`OnePatchPerFileBatchSampler` + ``RandomCrop``.
* ``flatten=True``  -> :class:`FileChunkedSampler` (MLP pixel-wise mode).
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import yaml


# Shared NPZ field plumbing used by the fixture (see conftest.tiny_npz_dir).
_RFT_KWARGS = {
    "fields_to_read": {"B": True, "E": True},
    "request_features": ["Bx", "By", "Bz"],
    "request_targets": ["Ex", "Ey", "Ez"],
    "verbose": False,
}


def _base_cfg(tiny_npz_dir: pathlib.Path, output_dir: pathlib.Path) -> dict:
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
            "loading_mode": "lazy_npz",
            "sample_cache_size": 1,
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


def _write(tmp_path: pathlib.Path, cfg: dict) -> pathlib.Path:
    p = tmp_path / "fit_lazy.yaml"
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


class TestCLIFitLazyNPZ:
    """End-to-end ``fit`` on the tiny on-disk NPZ fixture."""

    def test_fit_flatten_false_fcnn(self, tiny_npz_dir, monkeypatch):
        """FCNN + RandomCrop patch_dim path via OnePatchPerFileBatchSampler."""
        output_dir = tiny_npz_dir / "out_fcnn"
        output_dir.mkdir()
        cfg = _base_cfg(tiny_npz_dir, output_dir)
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
        # OnePatchPerFileBatchSampler requires batch_size <= num_files (5 train).
        cfg["data"]["batch_size"] = 2

        _run_cli(_write(tiny_npz_dir, cfg), monkeypatch)

        assert (output_dir / "X.pkl").exists(), "Feature norm file not created"
        assert (output_dir / "y.pkl").exists(), "Target norm file not created"

    def test_fit_flatten_true_mlp(self, tiny_npz_dir, monkeypatch):
        """MLP pixel-wise path via FileChunkedSampler."""
        output_dir = tiny_npz_dir / "out_mlp"
        output_dir.mkdir()
        cfg = _base_cfg(tiny_npz_dir, output_dir)
        cfg["model"]["network"] = {
            "class_path": "closure.models.MLP",
            "init_args": {
                "feature_dims": [3, 8, 3],
                "activations": ["ReLU", None],
            },
        }
        cfg["data"]["flatten"] = True
        cfg["data"]["chunk_window"] = 2
        cfg["data"]["sample_cache_size"] = 2
        cfg["data"]["batch_size"] = 8

        _run_cli(_write(tiny_npz_dir, cfg), monkeypatch)

        assert (output_dir / "X.pkl").exists(), "Feature norm file not created"
        assert (output_dir / "y.pkl").exists(), "Target norm file not created"
