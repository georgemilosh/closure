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
import sys
from pathlib import Path
from typing import Any

import yaml
from lightning.pytorch.cli import LightningCLI

from closure.module import ClosureLitModule
from closure.datamodule import ClosureDataModule


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
    """Resolve logger directory as ``save_dir/name/version`` when explicit.

    Important: if ``version`` is auto-assigned by Lightning (None), we must not
    pre-create a guessed ``version_*`` directory here because that can shift the
    version selected later by the logger itself.
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

    # Only route logs to a version directory when the version is explicit.
    if version is None:
        return None

    root = Path(save_dir).expanduser().resolve()
    version_dir = root / name / version
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


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


def _configure_python_logging(argv: list[str]) -> None:
    """Configure timestamped console/file logging with rank metadata."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_closure_configured", False):
        return

    _install_rank_record_factory()
    log_file = _resolve_log_file_path(argv)
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
    logging.getLogger(__name__).info("Logging initialized -> %s", log_file)


def main():
    """Launch Lightning CLI."""
    _configure_python_logging(sys.argv[1:])
    LightningCLI(
        ClosureLitModule,
        ClosureDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
