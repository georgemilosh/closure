"""Tests for closure.utilities."""

from __future__ import annotations

import subprocess

from closure import utilities


class TestBasicHelpers:
    def test_species_to_list(self):
        assert utilities.species_to_list(["Bx", "Jz_e"]) == ["Bx", ["Jz", "e"]]

    def test_append_index_to_duplicates(self):
        values = ["e", "i", "e", None, "i"]
        assert utilities.append_index_to_duplicates(values) == ["e1", "i1", "e2", None, "i2"]

    def test_get_duplicate_indices(self):
        values = ["e", "i", "e", None, "i"]
        assert utilities.get_duplicate_indices(values) == {"e": [0, 2], "i": [1, 4]}

    def test_get_git_revision_hash(self):
        expected = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        assert utilities.get_git_revision_hash() == expected


class TestCompatibilityWrappers:
    def test_plasma_wrapper_resolves(self):
        assert utilities.highdiff is not None
        assert utilities.get_Ohm is not None
        assert callable(utilities.vector_spectrum_2D)

    def test_unnormalize_alias(self):
        assert utilities.unnormalize_output is utilities.pred_unnormalize
