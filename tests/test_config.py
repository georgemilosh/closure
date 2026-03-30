"""Tests for closure.config — TrainerConfig, load_paths, load_config, set_nested_config."""

import json
import os
import pytest
from closure.config import TrainerConfig, load_paths, load_config, set_nested_config


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.work_dir is None
        assert cfg.log_name == "training.log"
        assert cfg.log_level == 20
        assert cfg.force is False
        assert cfg.mode_test is False
        assert cfg.world_size is None

    def test_custom_values(self):
        cfg = TrainerConfig(work_dir="/tmp/test", force=True, world_size=4, rank=1)
        assert cfg.work_dir == "/tmp/test"
        assert cfg.force is True
        assert cfg.world_size == 4
        assert cfg.rank == 1


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


class TestLoadConfig:
    def test_empty_files(self, tmp_path):
        cfg = load_config(
            config_file=str(tmp_path / "missing.json"),
            paths_file=str(tmp_path / "missing.yaml"),
        )
        assert isinstance(cfg, TrainerConfig)
        assert cfg.work_dir == "./outputs"

    def test_from_json(self, tmp_path):
        config_json = tmp_path / "config.json"
        config_json.write_text(json.dumps({
            "work_dir": "/test/work",
            "log_name": "test.log",
            "force": True,
            "num_workers": 8,
        }))
        cfg = load_config(config_file=str(config_json), paths_file=str(tmp_path / "x.yaml"))
        assert cfg.work_dir == "/test/work"
        assert cfg.log_name == "test.log"
        assert cfg.force is True
        assert cfg.num_workers == 8

    def test_paths_yaml_fallback(self, tmp_path):
        """work_dir from paths.yaml used when config.json has null."""
        config_json = tmp_path / "config.json"
        config_json.write_text(json.dumps({"work_dir": None, "force": False}))
        paths_yaml = tmp_path / "paths.yaml"
        paths_yaml.write_text("work_dir: /from/yaml\n")
        cfg = load_config(str(config_json), str(paths_yaml))
        assert cfg.work_dir == "/from/yaml"

    def test_unknown_fields_ignored(self, tmp_path):
        config_json = tmp_path / "config.json"
        config_json.write_text(json.dumps({"work_dir": "/w", "unknown_field": 42}))
        cfg = load_config(str(config_json), str(tmp_path / "x.yaml"))
        assert cfg.work_dir == "/w"
        assert not hasattr(cfg, "unknown_field")


class TestSetNestedConfig:
    def test_simple_key(self):
        cfg = {}
        set_nested_config(cfg, "lr", "0.001")
        assert cfg["lr"] == 0.001

    def test_nested_key(self):
        cfg = {}
        set_nested_config(cfg, "a.b.c", "123")
        assert cfg["a"]["b"]["c"] == 123

    def test_list_value(self):
        cfg = {}
        set_nested_config(cfg, "channels", "[10,64,16,6]")
        assert cfg["channels"] == [10, 64, 16, 6]

    def test_float_list(self):
        cfg = {}
        set_nested_config(cfg, "vals", "[1.1, 2.2]")
        assert cfg["vals"] == [1.1, 2.2]

    def test_string_list(self):
        cfg = {}
        set_nested_config(cfg, "act", "[ReLU,ReLU,ReLU]")
        assert cfg["act"] == ["ReLU", "ReLU", "ReLU"]

    def test_none_value(self):
        cfg = {}
        set_nested_config(cfg, "x", "None")
        assert cfg["x"] is None

    def test_non_string_value(self):
        cfg = {}
        set_nested_config(cfg, "x", 42)
        assert cfg["x"] == 42
