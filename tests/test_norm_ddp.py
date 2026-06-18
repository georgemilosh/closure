"""DDP-safety unit tests for ``_prepare_normalization_params``.

We don't spawn real ranks; instead we monkeypatch
``torch.distributed`` to make the dataset *believe* it lives inside a DDP
group, and verify that:

* Rank 0 computes and writes ``X.pkl`` / ``y.pkl`` then hits a barrier.
* Non-zero ranks skip the streaming compute, wait on the barrier, and
  load the norm file from disk.
* The single-process (no DDP) path still works.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pytest
import torch

from closure.datasets import LazyNPZDataFrameDataset


RFT_KWARGS = {
    "fields_to_read": {"B": True, "E": True},
    "request_features": ["Bx", "By", "Bz"],
    "request_targets": ["Ex", "Ey", "Ez"],
    "verbose": False,
}


class _FakeDist:
    """Minimal stand-in for ``torch.distributed`` controlled per-test."""

    def __init__(self, rank: int):
        self.rank = rank
        self.barrier_calls = 0
        self.initialized = True

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def get_rank(self) -> int:
        return self.rank

    def barrier(self) -> None:
        self.barrier_calls += 1


def _kwargs(data_folder, norm_folder, samples_file):
    return dict(
        data_folder=str(data_folder),
        norm_folder=str(norm_folder),
        samples_file=str(samples_file),
        datalabel="train",
        flatten=True,
        scaler_features=True,
        scaler_targets=True,
        read_features_targets_kwargs=RFT_KWARGS,
        sample_cache_size=2,
    )


class TestNormDDPBarrier:
    def test_rank0_computes_writes_and_barriers(self, tiny_npz_dir, monkeypatch):
        norm = tiny_npz_dir / "norm_rank0"
        fake = _FakeDist(rank=0)
        monkeypatch.setattr(torch, "distributed", fake)

        ds = LazyNPZDataFrameDataset(
            **_kwargs(tiny_npz_dir, norm, tiny_npz_dir / "train.csv")
        )

        assert (norm / "X.pkl").exists()
        assert (norm / "y.pkl").exists()
        # One barrier per scaler call (features + targets).
        assert fake.barrier_calls == 2
        # Rank 0 actually computed: arrays are finite and non-trivial.
        assert np.isfinite(ds.features_mean).all()
        assert (ds.features_std > 0).any()

    def test_non_zero_rank_loads_after_barrier(self, tiny_npz_dir, monkeypatch):
        # Pre-seed the norm dir as if rank 0 had already produced it.
        norm = tiny_npz_dir / "norm_rank1"
        rank0_dist = _FakeDist(rank=0)
        monkeypatch.setattr(torch, "distributed", rank0_dist)
        seeded = LazyNPZDataFrameDataset(
            **_kwargs(tiny_npz_dir, norm, tiny_npz_dir / "train.csv")
        )
        assert (norm / "X.pkl").exists()
        del seeded

        # Now wipe the file's existence check by re-pointing to a *fresh*
        # norm dir for the non-rank-0 build, then simulate that rank 0
        # writes the file *between* the file-exists check and the barrier.
        norm2 = tiny_npz_dir / "norm_rank1_b"
        norm2.mkdir()

        x_payload = joblib.load(norm / "X.pkl")
        y_payload = joblib.load(norm / "y.pkl")

        fake = _FakeDist(rank=1)

        # Simulate rank-0 writing exactly one norm file per barrier (features
        # then targets), matching the real two-call sequence.
        remaining = [("X.pkl", x_payload), ("y.pkl", y_payload)]

        def write_on_barrier():
            name, payload = remaining.pop(0)
            joblib.dump(payload, norm2 / name)
            fake.barrier_calls += 1

        fake.barrier = write_on_barrier  # type: ignore[assignment]
        monkeypatch.setattr(torch, "distributed", fake)

        # Patch streaming compute: rank != 0 must NOT call it.
        compute_calls = {"n": 0}
        orig_compute = LazyNPZDataFrameDataset._compute_streaming_normalization_params

        def tracked(self, data_type):
            compute_calls["n"] += 1
            return orig_compute(self, data_type)

        monkeypatch.setattr(
            LazyNPZDataFrameDataset,
            "_compute_streaming_normalization_params",
            tracked,
        )

        ds = LazyNPZDataFrameDataset(
            **_kwargs(tiny_npz_dir, norm2, tiny_npz_dir / "train.csv")
        )

        assert compute_calls["n"] == 0, "non-zero rank must not run streaming compute"
        assert fake.barrier_calls == 2
        # Loaded values match what rank 0 produced.
        np.testing.assert_allclose(ds.features_mean, x_payload[0])
        np.testing.assert_allclose(ds.targets_mean, y_payload[0])

    def test_single_process_path_unchanged(self, tiny_npz_dir, monkeypatch):
        norm = tiny_npz_dir / "norm_single"

        class _NotInit:
            def is_available(self):
                return True

            def is_initialized(self):
                return False

            def get_rank(self):  # should never be called
                raise AssertionError("get_rank called when DDP not initialized")

            def barrier(self):  # should never be called
                raise AssertionError("barrier called when DDP not initialized")

        monkeypatch.setattr(torch, "distributed", _NotInit())

        ds = LazyNPZDataFrameDataset(
            **_kwargs(tiny_npz_dir, norm, tiny_npz_dir / "train.csv")
        )

        assert (norm / "X.pkl").exists()
        assert (norm / "y.pkl").exists()
        assert np.isfinite(ds.features_mean).all()
