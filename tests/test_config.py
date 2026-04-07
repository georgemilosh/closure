"""Tests for closure.config — load_paths."""

import pytest
from closure.config import load_paths


class TestLoadPaths:
    def test_defaults_when_no_file(self, tmp_path):
        paths = load_paths(str(tmp_path / "nonexistent.yaml"))
        assert paths["work_dir"] == "./outputs"
        assert paths["data_dir"] == "./data"

    def test_loads_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "paths.yaml"
        yaml_file.write_text("work_dir: /my/work\ndata_dir: /my/data\n")
        paths = load_paths(str(yaml_file))
        assert paths["work_dir"] == "/my/work"
        assert paths["data_dir"] == "/my/data"

    def test_partial_override(self, tmp_path):
        yaml_file = tmp_path / "paths.yaml"
        yaml_file.write_text("work_dir: /custom\n")
        paths = load_paths(str(yaml_file))
        assert paths["work_dir"] == "/custom"
        assert paths["data_dir"] == "./data"  # default preserved
