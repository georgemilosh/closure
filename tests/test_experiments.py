"""Tests for closure.experiments discovery helpers."""

from __future__ import annotations

import pathlib

from closure import experiments as ex


def _make_run(root: pathlib.Path, name: str, *, with_output: bool = True) -> pathlib.Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    if with_output:
        (run / "iPIC3D-Fields_000000.npz").write_bytes(b"")
        (run / "iPIC3D-Fields_000500.npz").write_bytes(b"")
    return run


class TestHasOutputData:
    def test_true_when_field_files_present(self, tmp_path):
        run = _make_run(tmp_path, "run_a")
        assert ex.has_output_data(run) is True

    def test_false_for_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert ex.has_output_data(empty) is False

    def test_false_for_missing_dir(self, tmp_path):
        assert ex.has_output_data(tmp_path / "nope") is False


class TestDiscoverExperiments:
    def test_finds_immediate_children_with_output(self, tmp_path):
        _make_run(tmp_path, "run_a")
        _make_run(tmp_path, "run_b")
        _make_run(tmp_path, "no_output", with_output=False)
        assert ex.discover_experiments(tmp_path) == ["run_a", "run_b"]

    def test_recurses_into_nested_runs(self, tmp_path):
        _make_run(tmp_path, "campaign/R0/iso_GEM")
        _make_run(tmp_path, "campaign/R1/iso_GEM")
        found = ex.discover_experiments(tmp_path)
        assert found == ["campaign/R0/iso_GEM", "campaign/R1/iso_GEM"]

    def test_run_folder_is_a_leaf_products_not_descended(self, tmp_path):
        run = _make_run(tmp_path, "run_a")
        # A nested products dir that also happens to hold field-like files must
        # not be reported as a separate experiment.
        _make_run(run / "products", "sub", with_output=True)
        assert ex.discover_experiments(tmp_path) == ["run_a"]

    def test_non_recursive_ignores_nested(self, tmp_path):
        _make_run(tmp_path, "campaign/R0/iso_GEM")
        assert ex.discover_experiments(tmp_path, recursive=False) == []

    def test_max_depth_limits_descent(self, tmp_path):
        _make_run(tmp_path, "a/b/c/run")
        assert ex.discover_experiments(tmp_path, max_depth=2) == []
        assert ex.discover_experiments(tmp_path, max_depth=4) == ["a/b/c/run"]

    def test_missing_root_returns_empty(self, tmp_path):
        assert ex.discover_experiments(tmp_path / "nope") == []


class TestResolveExperiments:
    def test_explicit_list_passthrough(self, tmp_path):
        _make_run(tmp_path, "run_a")
        assert ex.resolve_experiments(["x", "y"], tmp_path) == ["x", "y"]

    def test_single_string_wrapped(self, tmp_path):
        assert ex.resolve_experiments("solo", tmp_path) == ["solo"]

    def test_none_triggers_discovery(self, tmp_path):
        _make_run(tmp_path, "run_a")
        _make_run(tmp_path, "run_b")
        assert ex.resolve_experiments(None, tmp_path) == ["run_a", "run_b"]

    def test_empty_list_triggers_discovery(self, tmp_path):
        _make_run(tmp_path, "run_a")
        assert ex.resolve_experiments([], tmp_path) == ["run_a"]
