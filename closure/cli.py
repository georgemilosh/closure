"""
cli.py — LightningCLI entry point for closure.

Provides a ``main()`` function that launches ``LightningCLI`` with
:class:`~closure.module.ClosureLitModule` and
:class:`~closure.datamodule.ClosureDataModule`.

Usage::

    closure-train fit --config configs/default.yaml
    closure-train validate --config configs/default.yaml
    closure-train test --config configs/default.yaml
    closure-train predict --config configs/default.yaml
"""

from __future__ import annotations

__all__ = ["main"]

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger

from closure.module import ClosureLitModule
from closure.datamodule import ClosureDataModule


def _run_git_command(repo_dir: Path, *args: str) -> str | None:
    """Run ``git`` command in *repo_dir* and return stripped stdout.

    Returns ``None`` when git is unavailable, the directory is not a repo,
    or the command fails for any reason.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None

    value = completed.stdout.strip()
    return value or None


def _get_git_revision_info() -> tuple[str | None, str | None]:
    """Return ``(branch, commit)`` for the current repository when available."""
    repo_dir = Path(__file__).resolve().parents[1]
    branch = _run_git_command(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _run_git_command(repo_dir, "rev-parse", "HEAD")
    return branch, commit


def _get_rank_from_env(name: str) -> int:
    """Return integer rank from env var, or -1 when missing/invalid."""
    value = os.getenv(name)
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _install_rank_record_factory() -> None:
    """Inject rank fields into all log records for formatter compatibility."""
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.global_rank = _get_rank_from_env("RANK")
        record.local_rank = _get_rank_from_env("LOCAL_RANK")
        return record

    logging.setLogRecordFactory(record_factory)


def _parse_default_root_dir_from_cli(argv: list[str]) -> str | None:
    """Extract ``trainer.default_root_dir`` from CLI args when provided."""
    key = "--trainer.default_root_dir"
    for i, arg in enumerate(argv):
        if arg.startswith(f"{key}="):
            return arg.split("=", 1)[1]
        if arg == key and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _parse_key_from_cli(argv: list[str], key: str) -> str | None:
    """Extract a CLI value for ``--key=value`` or ``--key value`` forms."""
    flag = f"--{key}"
    for i, arg in enumerate(argv):
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _parse_config_path_from_cli(argv: list[str]) -> Path | None:
    """Extract ``--config`` path from CLI args when provided."""
    key = "--config"
    for i, arg in enumerate(argv):
        if arg.startswith(f"{key}="):
            return Path(arg.split("=", 1)[1])
        if arg == key and i + 1 < len(argv):
            return Path(argv[i + 1])
    return None


def _default_root_dir_from_config(config_path: Path | None) -> str | None:
    """Read ``trainer.default_root_dir`` from YAML config if available."""
    if config_path is None or not config_path.exists():
        return None
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return None
    trainer_cfg = cfg.get("trainer", {}) if isinstance(cfg, dict) else {}
    if not isinstance(trainer_cfg, dict):
        return None
    value = trainer_cfg.get("default_root_dir")
    return str(value) if value else None


def _load_config(config_path: Path | None) -> dict[str, Any]:
    """Load YAML config into a dictionary, returning {} on failure."""
    if config_path is None or not config_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _logger_dir_from_config_and_cli(argv: list[str]) -> Path | None:
    """Resolve CSVLogger directory as the same log_dir Lightning will use.

    When ``version`` is implicit, CSVLogger can still expose the eventual
    ``version_*`` path without creating it, which keeps ``closure.log`` inside
    the per-run directory without shifting the version counter.
    """
    cfg_path = _parse_config_path_from_cli(argv)
    cfg = _load_config(cfg_path)

    trainer_cfg = cfg.get("trainer", {}) if isinstance(cfg, dict) else {}
    if not isinstance(trainer_cfg, dict):
        trainer_cfg = {}

    logger_cfg = trainer_cfg.get("logger", {})
    if not isinstance(logger_cfg, dict):
        logger_cfg = {}

    init_args = logger_cfg.get("init_args", {})
    if not isinstance(init_args, dict):
        init_args = {}

    save_dir = _parse_key_from_cli(argv, "trainer.logger.init_args.save_dir")
    if save_dir is None:
        cfg_save_dir = init_args.get("save_dir")
        save_dir = str(cfg_save_dir) if cfg_save_dir else None

    name = _parse_key_from_cli(argv, "trainer.logger.init_args.name")
    if name is None:
        cfg_name = init_args.get("name")
        name = str(cfg_name) if cfg_name else "lightning_logs"

    version = _parse_key_from_cli(argv, "trainer.logger.init_args.version")
    if version is None:
        cfg_version = init_args.get("version")
        version = str(cfg_version) if cfg_version is not None else None

    if save_dir is None:
        return None

    if logger_cfg.get("class_path") != "lightning.pytorch.loggers.CSVLogger":
        return None

    logger = CSVLogger(
        save_dir=str(Path(save_dir).expanduser().resolve()),
        name=name,
        version=version,
    )
    return Path(logger.log_dir).resolve()


def _resolve_log_file_path(argv: list[str]) -> Path:
    """Compute destination for ``closure.log``.

    Precedence:
    1) ``trainer.logger.init_args.save_dir/name/version`` (CLI/config)
    2) ``--trainer.default_root_dir`` CLI override
    3) ``trainer.default_root_dir`` from ``--config`` YAML
    4) current working directory
    """
    logger_dir = _logger_dir_from_config_and_cli(argv)
    if logger_dir is not None:
        root = logger_dir
    else:
        root_dir = _parse_default_root_dir_from_cli(argv)
        if root_dir is None:
            cfg_path = _parse_config_path_from_cli(argv)
            root_dir = _default_root_dir_from_config(cfg_path)
        if root_dir is None:
            root_dir = "."
        root = Path(root_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

    # Keep rank-separated files for distributed runs and a rank-0 default path.
    global_rank = _get_rank_from_env("RANK")
    if global_rank <= 0:
        return root / "closure.log"
    return root / f"closure_rank{global_rank}.log"


def _infer_csvlogger_save_dir_default(argv: list[str]) -> str | None:
    """Infer fallback ``save_dir`` for CSVLogger from ``default_root_dir``.

    Returns a path only when:
    - logger class is explicitly ``lightning.pytorch.loggers.CSVLogger``
    - ``trainer.logger.init_args.save_dir`` is not provided via CLI/config
    - ``trainer.default_root_dir`` is available via CLI/config
    """
    explicit_save_dir = _parse_key_from_cli(argv, "trainer.logger.init_args.save_dir")
    if explicit_save_dir:
        return None

    cfg_path = _parse_config_path_from_cli(argv)
    cfg = _load_config(cfg_path)

    trainer_cfg = cfg.get("trainer", {}) if isinstance(cfg, dict) else {}
    if not isinstance(trainer_cfg, dict):
        return None

    logger_cfg = trainer_cfg.get("logger", {})
    if not isinstance(logger_cfg, dict):
        return None

    class_path = logger_cfg.get("class_path")
    if class_path != "lightning.pytorch.loggers.CSVLogger":
        return None

    init_args = logger_cfg.get("init_args", {})
    if not isinstance(init_args, dict):
        init_args = {}

    cfg_save_dir = init_args.get("save_dir")
    if cfg_save_dir:
        return None

    root_dir = _parse_default_root_dir_from_cli(argv)
    if root_dir is None:
        root_dir_value = trainer_cfg.get("default_root_dir")
        root_dir = str(root_dir_value) if root_dir_value else None

    return root_dir


def _infer_csvlogger_version_default(argv: list[str]) -> str | None:
    """Infer the implicit CSVLogger version directory name.

    Returns a version folder name like ``version_0`` only when:
    - logger class is explicitly ``lightning.pytorch.loggers.CSVLogger``
    - ``trainer.logger.init_args.version`` is not provided via CLI/config
    - ``trainer.logger.init_args.save_dir`` is available via CLI/config/inference
    """
    explicit_version = _parse_key_from_cli(argv, "trainer.logger.init_args.version")
    if explicit_version is not None:
        return None

    cfg_path = _parse_config_path_from_cli(argv)
    cfg = _load_config(cfg_path)

    trainer_cfg = cfg.get("trainer", {}) if isinstance(cfg, dict) else {}
    if not isinstance(trainer_cfg, dict):
        return None

    logger_cfg = trainer_cfg.get("logger", {})
    if not isinstance(logger_cfg, dict):
        return None

    class_path = logger_cfg.get("class_path")
    if class_path != "lightning.pytorch.loggers.CSVLogger":
        return None

    init_args = logger_cfg.get("init_args", {})
    if not isinstance(init_args, dict):
        init_args = {}

    cfg_version = init_args.get("version")
    if cfg_version is not None:
        return None

    save_dir = _parse_key_from_cli(argv, "trainer.logger.init_args.save_dir")
    if save_dir is None:
        cfg_save_dir = init_args.get("save_dir")
        save_dir = str(cfg_save_dir) if cfg_save_dir else None
    if save_dir is None:
        return None

    name = _parse_key_from_cli(argv, "trainer.logger.init_args.name")
    if name is None:
        cfg_name = init_args.get("name")
        name = str(cfg_name) if cfg_name else "lightning_logs"

    logger = CSVLogger(
        save_dir=str(Path(save_dir).expanduser().resolve()),
        name=name,
        version=None,
    )
    return Path(logger.log_dir).name


def _infer_norm_folder_default(argv: list[str]) -> str | None:
    """Infer fallback ``data.norm_folder`` from ``trainer.default_root_dir``.

    Returns a path only when:
    - ``data.norm_folder`` is not provided via CLI/config
    - ``trainer.default_root_dir`` is available via CLI/config
    """
    explicit_norm_folder = _parse_key_from_cli(argv, "data.norm_folder")
    if explicit_norm_folder:
        return None

    cfg_path = _parse_config_path_from_cli(argv)
    cfg = _load_config(cfg_path)

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    cfg_norm_folder = data_cfg.get("norm_folder")
    if cfg_norm_folder:
        return None

    trainer_cfg = cfg.get("trainer", {}) if isinstance(cfg, dict) else {}
    if not isinstance(trainer_cfg, dict):
        return None

    root_dir = _parse_default_root_dir_from_cli(argv)
    if root_dir is None:
        root_dir_value = trainer_cfg.get("default_root_dir")
        root_dir = str(root_dir_value) if root_dir_value else None

    return root_dir


def _configure_python_logging(argv: list[str]) -> None:
    """Configure timestamped console/file logging with rank metadata."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_closure_configured", False):
        return

    _install_rank_record_factory()
    log_file = _resolve_log_file_path(argv)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = (
        "%(asctime)s %(levelname)s [%(name)s]"
        " [rank=%(global_rank)s local_rank=%(local_rank)s] %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file)
    for handler in (stream_handler, file_handler):
        handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    root_logger._closure_configured = True
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized -> %s", log_file)

    branch, commit = _get_git_revision_info()
    if branch and commit:
        logger.info("Git revision -> branch=%s commit=%s", branch, commit)
    else:
        logger.info("Git revision -> unavailable")


def _extract_cli_overrides(argv: list[str]) -> list[str]:
    """Return user-provided ``--key value`` / ``--key=value`` overrides.

    Excludes the ``--config`` argument because the manifest stores it separately.
    """
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]

        # Skip --config (both forms).
        if arg.startswith("--config="):
            i += 1
            continue
        if arg == "--config":
            i += 2
            continue

        if arg.startswith("--"):
            if "=" in arg:
                overrides.append(arg)
            else:
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    overrides.append(f"{arg} {argv[i + 1]}")
                    i += 1
                else:
                    overrides.append(arg)
        i += 1

    return overrides


def _to_yaml_safe(value: Any) -> Any:
    """Recursively convert values to YAML-safe built-in Python types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _to_yaml_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_yaml_safe(v) for v in value]

    # jsonargparse Path-like objects expose an absolute path via __fspath__.
    fspath = getattr(value, "__fspath__", None)
    if callable(fspath):
        try:
            return str(fspath())
        except Exception:
            pass

    # Fallback: preserve information using string representation.
    return str(value)


class ClosureLightningCLI(LightningCLI):
    """Project LightningCLI with reproducibility manifest support."""

    def __init__(
        self,
        *args,
        user_argv: list[str] | None = None,
        inferred_overrides: list[str] | None = None,
        **kwargs,
    ) -> None:
        self._user_argv = list(user_argv or [])
        self._inferred_overrides = list(inferred_overrides or [])
        self._manifest_written = False
        super().__init__(*args, **kwargs)

    def _resolved_config_dict(self) -> dict[str, Any]:
        """Get fully resolved CLI config as a plain dictionary."""
        cfg = getattr(self, "config", None)
        if cfg is None:
            return {}
        if hasattr(cfg, "as_dict"):
            try:
                as_dict = cfg.as_dict()
                return as_dict if isinstance(as_dict, dict) else {}
            except Exception:
                pass
        try:
            dumped = self.parser.dump(cfg, skip_none=False, format="yaml")
            loaded = yaml.safe_load(dumped) or {}
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def _manifest_dir(self) -> Path:
        """Return destination directory for run manifests."""
        log_dir = getattr(self.trainer, "log_dir", None)
        if log_dir:
            manifest_dir = Path(log_dir)
        else:
            manifest_dir = Path(self.trainer.default_root_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        return manifest_dir

    def _write_repro_manifest(self, stage: str) -> None:
        """Persist reproducibility manifest once per run."""
        if self._manifest_written:
            return

        config_path = _parse_config_path_from_cli(self._user_argv)
        branch, commit = _get_git_revision_info()
        payload: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "command": " ".join([Path(sys.argv[0]).name, *self._user_argv]),
            "config_file": str(config_path) if config_path else None,
            "cli_overrides": _extract_cli_overrides(self._user_argv),
            "inferred_cli_overrides": self._inferred_overrides,
            "git": {"branch": branch, "commit": commit},
            "resolved_config": self._resolved_config_dict(),
        }

        manifest_path = self._manifest_dir() / "repro_manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(_to_yaml_safe(payload), sort_keys=False))
        logging.getLogger(__name__).info("Wrote reproducibility manifest -> %s", manifest_path)
        self._manifest_written = True

    def before_fit(self) -> None:
        self._write_repro_manifest(stage="fit")

    def before_validate(self) -> None:
        self._write_repro_manifest(stage="validate")

    def before_test(self) -> None:
        self._write_repro_manifest(stage="test")

    def before_predict(self) -> None:
        self._write_repro_manifest(stage="predict")


def main():
    """Launch Lightning CLI."""
    user_argv = sys.argv[1:]
    argv = list(user_argv)
    inferred_overrides: list[str] = []

    inferred_norm_folder = _infer_norm_folder_default(argv)
    if inferred_norm_folder:
        arg = f"--data.norm_folder={inferred_norm_folder}"
        argv = [*argv, arg]
        inferred_overrides.append(arg)

    inferred_save_dir = _infer_csvlogger_save_dir_default(argv)
    if inferred_save_dir:
        arg = f"--trainer.logger.init_args.save_dir={inferred_save_dir}"
        argv = [*argv, arg]
        inferred_overrides.append(arg)

    inferred_version = _infer_csvlogger_version_default(argv)
    if inferred_version:
        arg = f"--trainer.logger.init_args.version={inferred_version}"
        argv = [*argv, arg]
        inferred_overrides.append(arg)

    _configure_python_logging(argv)
    ClosureLightningCLI(
        ClosureLitModule,
        ClosureDataModule,
        user_argv=user_argv,
        inferred_overrides=inferred_overrides,
        args=argv,
        trainer_defaults={"precision": "32-true"},
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
