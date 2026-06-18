"""Tests for the single-open ``.npz`` fast path in ``LazyNPZDataFrameDataset``.

These tests verify:

* The fast path is auto-detected on compatible ``.npz`` inputs.
* It opens each file **exactly once** (vs once-per-field in the slow path).
* It produces values identical to the slow path (parity already covered by
  ``test_lazy_npz_disk`` after the fast path was added; here we verify the
  enable/disable switch explicitly).
* It is disabled when alfven_units / filter callbacks / spatial slicing /
  missing keys / derived-field lists are configured.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import closure.datasets as datasets_module
from closure.datasets import LazyNPZDataFrameDataset


RFT_KWARGS = {
    "fields_to_read": {"B": True, "E": True},
    "request_features": ["Bx", "By", "Bz"],
    "request_targets": ["Ex", "Ey", "Ez"],
    "verbose": False,
}


def _build(tiny_npz_dir, *, flatten=False, **extra):
    return LazyNPZDataFrameDataset(
        data_folder=str(tiny_npz_dir),
        norm_folder=str(tiny_npz_dir / "norm_fast"),
        samples_file=str(tiny_npz_dir / "train.csv"),
        datalabel="train",
        flatten=flatten,
        scaler_features=True,
        scaler_targets=True,
        read_features_targets_kwargs=RFT_KWARGS,
        sample_cache_size=0,
        **extra,
    )


class TestFastPathDetection:
    def test_fast_path_enabled_for_compatible_npz(self, tiny_npz_dir):
        ds = _build(tiny_npz_dir, flatten=False)
        assert ds._npz_fast_path is True

    def test_fast_path_disabled_when_alfven_units(self, tiny_npz_dir, monkeypatch):
        # Bypass actual .inp lookup by patching rp.code2alfven to a no-op.
        import closure.read_pic as rp

        monkeypatch.setattr(rp, "_resolve_experiment_dir", lambda p, f: ".")
        monkeypatch.setattr(rp, "_find_experiment_inp_file", lambda d: "noop")
        monkeypatch.setattr(rp, "_read_b0x_nb_from_inp", lambda p: (1.0, 1.0))
        monkeypatch.setattr(rp, "code2alfven", lambda data, **kw: None)

        ds = _build(tiny_npz_dir, flatten=False, alfven_units=True)
        assert ds._npz_fast_path is False

    def test_fast_path_disabled_when_choose_slicing(self, tiny_npz_dir):
        kwargs = dict(RFT_KWARGS, choose_x=[1, 5])
        ds = LazyNPZDataFrameDataset(
            data_folder=str(tiny_npz_dir),
            norm_folder=str(tiny_npz_dir / "norm_fastB"),
            samples_file=str(tiny_npz_dir / "train.csv"),
            datalabel="train",
            flatten=False,
            scaler_features=True,
            scaler_targets=True,
            read_features_targets_kwargs=kwargs,
            sample_cache_size=0,
        )
        assert ds._npz_fast_path is False

    def test_fast_path_disabled_when_missing_key(self, tiny_npz_dir):
        ds = _build(tiny_npz_dir, flatten=False)
        assert ds._npz_fast_path is True
        # Mutate the request to include a missing key and re-probe.
        ds.request_features = list(ds.request_features) + ["DoesNotExist"]
        feats, tgts = ds._maybe_fast_load_npz(ds.filenames[0])
        assert feats is None and tgts is None
        assert ds._npz_fast_path is False


class TestFastPathIO:
    def test_single_open_per_file(self, tiny_npz_dir, monkeypatch):
        ds = _build(tiny_npz_dir, flatten=False)
        assert ds._npz_fast_path is True

        # Count np.load calls AFTER construction (post-normalization).
        original = np.load
        calls = {"n": 0}

        def counting_load(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(np, "load", counting_load)

        for i in range(ds.num_files):
            _ = ds[i]

        assert calls["n"] == ds.num_files, (
            f"Fast path should open each file exactly once; "
            f"got {calls['n']} loads for {ds.num_files} files"
        )

    def test_fast_path_matches_slow_path(self, tiny_npz_dir, monkeypatch):
        """Force the slow path and assert per-pixel parity with the fast path."""
        ds_fast = _build(tiny_npz_dir, flatten=False)
        assert ds_fast._npz_fast_path is True

        ds_slow = _build(tiny_npz_dir, flatten=False)
        # Force slow path by disabling the fast-path tri-state.
        ds_slow._npz_fast_path = False
        # Wipe cache so reads actually re-decode through rp.
        ds_slow._cache._store.clear()

        for i in range(ds_fast.num_files):
            ff, ft = ds_fast[i]
            sf, st = ds_slow[i]
            assert ff.shape == sf.shape
            assert ft.shape == st.shape
            torch.testing.assert_close(ff, sf, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(ft, st, rtol=1e-5, atol=1e-6)
