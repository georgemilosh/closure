"""End-to-end test for ``closure-train fit`` using the ecsim_tiny fixture."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest
import yaml

from closure.callbacks import TorchScriptCheckpointExportCallback
from closure.cli import (
    _extract_cli_overrides,
    _ensure_checkpoint_pt_export_callback,
    _get_git_revision_info,
    _infer_csvlogger_save_dir_default,
    _infer_csvlogger_version_default,
    _infer_norm_folder_default,
    _run_git_command,
    _resolve_log_file_path,
)


FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "ecsim_tiny"


def _write_fit_config(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a minimal YAML config that points at the ecsim_tiny fixture."""
    norm_dir = tmp_path / "norm"
    norm_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    cfg = {
        "seed_everything": 42,
        "model": {
            "network": {
                "class_path": "closure.models.FCNN",
                "init_args": {
                    "channels": [4, 8, 3],
                    "kernels": [3, 3],
                    "activations": ["ReLU", None],
                },
            },
            "criterion": "MSELoss",
            "optimizer": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0,
            "scheduler": None,
        },
        "data": {
            "data_folder": str(FIXTURES_DIR),
            "norm_folder": str(norm_dir),
            "train_samples_file": str(FIXTURES_DIR / "train.csv"),
            "val_samples_file": str(FIXTURES_DIR / "val.csv"),
            "test_samples_file": str(FIXTURES_DIR / "test.csv"),
            "batch_size": 2,
            "num_workers": 0,
            "flatten": False,
            "scaler_features": True,
            "scaler_targets": True,
            "features_dtype": "float32",
            "targets_dtype": "float32",
            "read_features_targets_kwargs": {
                "fields_to_read": {
                    "B": True,
                    "B_ext": False,
                    "E": False,
                    "E_ext": False,
                    "divB": False,
                    "rho": True,
                    "J": True,
                    "P": True,
                    "PI": False,
                    "Heat_flux": False,
                },
                "request_features": ["rho_e", "Bx", "By", "Bz"],
                "request_targets": ["Pxx_e", "Pyy_e", "Pzz_e"],
                "choose_species": ["e", None],
                "choose_x": [0, 16],
                "choose_y": [0, 16],
                "verbose": False,
            },
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

    config_path = tmp_path / "test_fit.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path


class TestCLIFit:
    """Run ``closure-train fit`` end-to-end on tiny fixture data."""

    def test_fit_completes(self, tmp_path, monkeypatch):
        """A fast_dev_run fit should complete without errors."""
        config_path = _write_fit_config(tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["closure-train", "fit", f"--config={config_path}"]
        )

        from closure.cli import main

        # LightningCLI calls sys.exit(0) on success in some versions;
        # catch SystemExit so the test doesn't abort.
        try:
            main()
        except SystemExit as exc:
            assert exc.code in (None, 0), f"CLI exited with code {exc.code}"

    def test_fit_produces_norm_files(self, tmp_path, monkeypatch):
        """Normalization stats (X.pkl, y.pkl) should be created in norm_folder."""
        config_path = _write_fit_config(tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["closure-train", "fit", f"--config={config_path}"]
        )

        from closure.cli import main

        try:
            main()
        except SystemExit:
            pass

        norm_dir = tmp_path / "norm"
        assert (norm_dir / "X.pkl").exists(), "Feature norm file not created"
        assert (norm_dir / "y.pkl").exists(), "Target norm file not created"


def test_log_file_goes_to_logger_version_dir(tmp_path):
    """closure.log should be placed under save_dir/name/version when configured."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "trainer": {
                    "default_root_dir": str(tmp_path / "fallback_root"),
                    "logger": {
                        "class_path": "lightning.pytorch.loggers.CSVLogger",
                        "init_args": {
                            "save_dir": str(tmp_path / "logs"),
                            "name": "CNN",
                            "version": "run_2",
                        },
                    },
                }
            }
        )
    )

    path = _resolve_log_file_path(["fit", f"--config={config_path}"])
    expected = tmp_path / "logs" / "CNN" / "run_2" / "closure.log"
    assert path == expected.resolve()


def test_log_file_falls_back_to_default_root_dir(tmp_path):
    """When logger save_dir is absent, default_root_dir remains the destination."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "trainer": {
                    "default_root_dir": str(tmp_path / "models"),
                    "logger": False,
                }
            }
        )
    )

    path = _resolve_log_file_path(["fit", f"--config={config_path}"])
    expected = tmp_path / "models" / "closure.log"
    assert path == expected.resolve()


def test_log_file_falls_back_when_logger_version_is_implicit(tmp_path):
    """Without explicit version, use the same version_* dir as CSVLogger."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "trainer": {
                    "default_root_dir": str(tmp_path / "models"),
                    "logger": {
                        "class_path": "lightning.pytorch.loggers.CSVLogger",
                        "init_args": {
                            "save_dir": str(tmp_path / "logs"),
                            "name": "CNN",
                        },
                    },
                }
            }
        )
    )

    path = _resolve_log_file_path(["fit", f"--config={config_path}"])
    expected = tmp_path / "logs" / "CNN" / "version_0" / "closure.log"
    assert path == expected.resolve()


def test_csvlogger_save_dir_defaults_to_default_root_dir(tmp_path):
    """Inject save_dir from default_root_dir when CSVLogger save_dir is omitted."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "trainer": {
                    "default_root_dir": str(tmp_path / "models"),
                    "logger": {
                        "class_path": "lightning.pytorch.loggers.CSVLogger",
                        "init_args": {
                            "name": "CNN",
                        },
                    },
                }
            }
        )
    )

    inferred = _infer_csvlogger_save_dir_default(["fit", f"--config={config_path}"])
    assert inferred == str(tmp_path / "models")


def test_csvlogger_save_dir_not_overridden_when_explicit(tmp_path):
    """Do not inject fallback when save_dir is already set in config."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "trainer": {
                    "default_root_dir": str(tmp_path / "models"),
                    "logger": {
                        "class_path": "lightning.pytorch.loggers.CSVLogger",
                        "init_args": {
                            "save_dir": str(tmp_path / "logs"),
                            "name": "CNN",
                        },
                    },
                }
            }
        )
    )

    inferred = _infer_csvlogger_save_dir_default(["fit", f"--config={config_path}"])
    assert inferred is None


def test_csvlogger_version_is_inferred_when_implicit(tmp_path):
    """Infer the implicit version_* folder so logging can target the run dir."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "trainer": {
                    "default_root_dir": str(tmp_path / "models"),
                    "logger": {
                        "class_path": "lightning.pytorch.loggers.CSVLogger",
                        "init_args": {
                            "save_dir": str(tmp_path / "logs"),
                            "name": "CNN",
                        },
                    },
                }
            }
        )
    )

    inferred = _infer_csvlogger_version_default(["fit", f"--config={config_path}"])
    assert inferred == "version_0"


def test_checkpoint_pt_export_callback_is_attached_once():
    trainer = SimpleNamespace(callbacks=[])

    _ensure_checkpoint_pt_export_callback(trainer)
    _ensure_checkpoint_pt_export_callback(trainer)

    matching = [cb for cb in trainer.callbacks if isinstance(cb, TorchScriptCheckpointExportCallback)]
    assert len(matching) == 1


def test_norm_folder_defaults_to_default_root_dir(tmp_path):
    """Inject norm_folder from default_root_dir when norm_folder is omitted."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "trainer": {
                    "default_root_dir": str(tmp_path / "models"),
                },
                "data": {
                    "data_folder": "ecsim/Harris/Le/",
                },
            }
        )
    )

    inferred = _infer_norm_folder_default(["fit", f"--config={config_path}"])
    assert inferred == str(tmp_path / "models")


def test_extract_cli_overrides_excludes_config_and_keeps_flags():
    """CLI override extraction should ignore --config and preserve override args."""
    argv = [
        "fit",
        "--config=configs/default.yaml",
        "--trainer.max_epochs=5",
        "--data.batch_size",
        "16",
        "--model.dropout",
        "0.1",
        "--trainer.enable_progress_bar",
        "false",
    ]
    overrides = _extract_cli_overrides(argv)
    assert "--config=configs/default.yaml" not in overrides
    assert "--trainer.max_epochs=5" in overrides
    assert "--data.batch_size 16" in overrides
    assert "--model.dropout 0.1" in overrides
    assert "--trainer.enable_progress_bar false" in overrides


def test_norm_folder_not_overridden_when_explicit(tmp_path):
    """Do not inject fallback when norm_folder is already set in config."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "trainer": {
                    "default_root_dir": str(tmp_path / "models"),
                },
                "data": {
                    "data_folder": "ecsim/Harris/Le/",
                    "norm_folder": str(tmp_path / "norm"),
                },
            }
        )
    )

    inferred = _infer_norm_folder_default(["fit", f"--config={config_path}"])
    assert inferred is None


def test_run_git_command_returns_stdout(monkeypatch, tmp_path):
    """Git command wrapper should return stripped stdout when successful."""

    def _fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="main\n")

    monkeypatch.setattr("closure.cli.subprocess.run", _fake_run)
    value = _run_git_command(tmp_path, "rev-parse", "--abbrev-ref", "HEAD")
    assert value == "main"


def test_run_git_command_returns_none_on_error(monkeypatch, tmp_path):
    """Git command wrapper should return None on non-zero exit status."""

    def _fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("closure.cli.subprocess.run", _fake_run)
    value = _run_git_command(tmp_path, "rev-parse", "HEAD")
    assert value is None


def test_get_git_revision_info_uses_git_commands(monkeypatch):
    """Branch and commit should come from helper git commands."""

    calls = []

    def _fake_run_git_command(_repo_dir, *args):
        calls.append(args)
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "feature/test"
        if args == ("rev-parse", "HEAD"):
            return "0123456789abcdef"
        return None

    monkeypatch.setattr("closure.cli._run_git_command", _fake_run_git_command)
    branch, commit = _get_git_revision_info()

    assert branch == "feature/test"
    assert commit == "0123456789abcdef"
    assert ("rev-parse", "--abbrev-ref", "HEAD") in calls
    assert ("rev-parse", "HEAD") in calls
