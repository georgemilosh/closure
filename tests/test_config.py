"""Tests for closure.config — load_paths."""

import os
import pytest
from closure.config import load_paths


class TestLoadPaths:
    def test_defaults_when_no_file(self, tmp_path):
        paths = load_paths(str(tmp_path / "nonexistent.yaml"))
        # Relative defaults are resolved against the directory of the paths file
        assert os.path.isabs(paths["work_dir"])
        assert os.path.isabs(paths["data_dir"])
        assert paths["work_dir"] == os.path.normpath(os.path.join(str(tmp_path), "outputs"))
        assert paths["data_dir"] == os.path.normpath(os.path.join(str(tmp_path), "data"))

    def test_loads_absolute_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "paths.yaml"
        yaml_file.write_text("work_dir: /my/work\ndata_dir: /my/data\n")
        paths = load_paths(str(yaml_file))
        assert paths["work_dir"] == "/my/work"
        assert paths["data_dir"] == "/my/data"

    def test_loads_relative_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "paths.yaml"
        yaml_file.write_text("work_dir: ./models\ndata_dir: ./data\n")
        paths = load_paths(str(yaml_file))
        assert paths["work_dir"] == os.path.normpath(os.path.join(str(tmp_path), "models"))
        assert paths["data_dir"] == os.path.normpath(os.path.join(str(tmp_path), "data"))

    def test_partial_override(self, tmp_path):
        yaml_file = tmp_path / "paths.yaml"
        yaml_file.write_text("work_dir: /custom\n")
        paths = load_paths(str(yaml_file))
        assert paths["work_dir"] == "/custom"
        # data_dir default is relative, resolved against yaml directory
        assert paths["data_dir"] == os.path.normpath(os.path.join(str(tmp_path), "data"))
