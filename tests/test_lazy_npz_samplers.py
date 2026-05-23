"""Unit tests for LazyNPZDataFrameDataset's batch samplers.

Covers:
  * OnePatchPerFileBatchSampler: distinct file indices per batch,
    correct length, reproducibility with set_epoch, raises on misuse.
  * FileChunkedSampler: full coverage of (file, pixel) space exactly once,
    window-bounded live-cache footprint, reproducibility.

End-to-end dataset behavior (parity with eager, cache hits, normalization
stats) is covered by ``tests/test_lazy_npz_disk.py`` using real npz files,
which is more reliable than the previous mock-based approach.
"""

from __future__ import annotations

import pytest

from closure.datasets import FileChunkedSampler, OnePatchPerFileBatchSampler


# ----------------------------------------------------------------------
# OnePatchPerFileBatchSampler
# ----------------------------------------------------------------------
class TestOnePatchPerFileBatchSampler:
    def test_distinct_files_per_batch(self):
        s = OnePatchPerFileBatchSampler(
            num_files=10, batch_size=4, oversample=2, drop_last=True, seed=0
        )
        batches = list(iter(s))
        for b in batches:
            assert len(b) == 4
            assert len(set(b)) == 4

    def test_length_matches_iter(self):
        s = OnePatchPerFileBatchSampler(
            num_files=10, batch_size=3, oversample=2, drop_last=True
        )
        batches = list(iter(s))
        # 10 // 3 = 3 batches per pass; 2 passes -> 6 batches.
        assert len(s) == 6
        assert len(batches) == len(s)

    def test_drop_last_false_keeps_remainder(self):
        s = OnePatchPerFileBatchSampler(
            num_files=10, batch_size=3, oversample=1, drop_last=False
        )
        batches = list(iter(s))
        assert len(batches) == 4
        assert len(batches[-1]) == 1

    def test_set_epoch_changes_order(self):
        s = OnePatchPerFileBatchSampler(
            num_files=8, batch_size=2, oversample=1, seed=123
        )
        s.set_epoch(0)
        b0 = list(iter(s))
        s.set_epoch(0)
        b0_again = list(iter(s))
        s.set_epoch(1)
        b1 = list(iter(s))
        assert b0 == b0_again
        assert b0 != b1

    def test_no_shuffle_is_deterministic_range(self):
        s = OnePatchPerFileBatchSampler(
            num_files=6, batch_size=2, oversample=1, shuffle=False
        )
        assert list(iter(s)) == [[0, 1], [2, 3], [4, 5]]

    def test_raises_when_batch_too_large(self):
        with pytest.raises(ValueError, match="incompatible"):
            OnePatchPerFileBatchSampler(num_files=4, batch_size=8)


# ----------------------------------------------------------------------
# FileChunkedSampler
# ----------------------------------------------------------------------
class TestFileChunkedSampler:
    def test_covers_all_pixels_once(self):
        s = FileChunkedSampler(num_files=5, pixels_per_file=6, window=2, seed=0)
        out = list(iter(s))
        assert len(out) == len(s) == 5 * 6
        assert sorted(out) == list(range(5 * 6))

    def test_window_one_processes_one_file_at_a_time(self):
        s = FileChunkedSampler(
            num_files=4, pixels_per_file=3, window=1, shuffle=False
        )
        out = list(iter(s))
        # With shuffle=False and window=1 file order is 0,1,2,3 and pixel
        # order is 0,1,2 within each file.
        assert out == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def test_window_bounds_live_cache_footprint(self):
        # FileChunkedSampler partitions files into non-overlapping groups
        # of ``window`` files. Within each group's W*ppf emitted indices,
        # exactly W distinct file_idx values appear.
        ppf = 4
        W = 2
        n = 6
        s = FileChunkedSampler(num_files=n, pixels_per_file=ppf, window=W, seed=0)
        emitted = list(iter(s))
        chunk = W * ppf
        for start in range(0, len(emitted), chunk):
            window_slice = emitted[start:start + chunk]
            file_ids = {idx // ppf for idx in window_slice}
            assert len(file_ids) <= W

    def test_set_epoch_changes_order(self):
        s = FileChunkedSampler(num_files=4, pixels_per_file=3, window=2, seed=99)
        s.set_epoch(0)
        a = list(iter(s))
        s.set_epoch(0)
        b = list(iter(s))
        s.set_epoch(1)
        c = list(iter(s))
        assert a == b
        assert a != c
