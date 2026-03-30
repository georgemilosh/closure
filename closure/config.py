"""
config.py — Configuration management for the closure package.

Provides TrainerConfig dataclass, YAML-based path loading,
JSON config loading, and nested config helpers.
"""

from __future__ import annotations

__all__ = ["TrainerConfig", "load_paths", "load_config", "set_nested_config"]

import ast
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class TrainerConfig:
    """Typed configuration for Trainer.

    Parameters
    ----------
    work_dir : str or None
        Directory to save training outputs.
    dataset_kwargs : dict or None
        Keyword arguments for creating dataset objects.
    load_data_kwargs : dict or None
        Keyword arguments for data loaders.
    model_kwargs : dict or None
        Keyword arguments for creating the model.
    device : str or None
        Device to use for training (e.g. ``"cpu"``, ``"cuda"``).
    mode_test : bool
        If True, skip loading train/val datasets (inference mode).
    log_name : str
        Name of the log file.
    log_level : int
        Logging level (e.g. ``logging.INFO``).
    num_workers : int or None
        Number of workers for data loading.
    force : bool
        Overwrite existing run directories if True.
    timing_name : str or None
        Name of the timing CSV file.  ``None`` disables timing.
    world_size : int or None
        Number of processes in distributed training.
    rank : int or None
        Global rank of the current process.
    gpus_per_node : int or None
        Number of GPUs per node.
    local_rank : int or None
        Local rank of the current process.
    """

    work_dir: Optional[str] = None
    dataset_kwargs: Optional[dict[str, Any]] = None
    load_data_kwargs: Optional[dict[str, Any]] = None
    model_kwargs: Optional[dict[str, Any]] = None
    device: Optional[str] = None
    mode_test: bool = False
    log_name: str = "training.log"
    log_level: int = 20  # logging.INFO
    num_workers: Optional[int] = None
    force: bool = False
    timing_name: Optional[str] = None
    # Distributed training
    world_size: Optional[int] = None
    rank: Optional[int] = None
    gpus_per_node: Optional[int] = None
    local_rank: Optional[int] = None


def load_paths(paths_file: str = "paths.yaml") -> dict[str, str]:
    """Load local paths from *paths.yaml*, falling back to defaults.

    Parameters
    ----------
    paths_file : str
        Path to the YAML file with ``work_dir`` / ``data_dir`` keys.

    Returns
    -------
    dict[str, str]
        Dictionary with at least ``work_dir`` and ``data_dir``.
    """
    defaults = {"work_dir": "./outputs", "data_dir": "./data"}
    if os.path.exists(paths_file):
        with open(paths_file, "r") as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
    return defaults


def load_config(
    config_file: str = "config.json",
    paths_file: str = "paths.yaml",
) -> TrainerConfig:
    """Build a :class:`TrainerConfig` from *config.json* + *paths.yaml*.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.
    paths_file : str
        Path to the YAML paths file.

    Returns
    -------
    TrainerConfig
        Fully populated configuration dataclass.
    """
    paths = load_paths(paths_file)

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            raw = json.load(f)
    else:
        raw = {}

    # If work_dir not in JSON, use paths.yaml value
    if raw.get("work_dir") is None:
        raw["work_dir"] = paths.get("work_dir", "./outputs")

    # Map JSON log_level string → int if needed
    log_level = raw.get("log_level")
    if isinstance(log_level, str):
        import logging as _logging

        raw["log_level"] = getattr(_logging, log_level, 20)

    # Only pass fields that TrainerConfig knows about
    valid_fields = {f.name for f in TrainerConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    return TrainerConfig(**filtered)


def set_nested_config(config: dict, key: str, value: Any) -> None:
    """Set a nested configuration value in a dictionary.

    Parameters
    ----------
    config : dict
        The configuration dictionary to update.
    key : str
        A dot-separated string specifying the nested key
        (e.g. ``"model_kwargs.optimizer_kwargs.lr"``).
    value : str or Any
        The value to set.  If a string, it will be converted to
        ``int``, ``float``, ``list``, or ``None`` when possible.

    Examples
    --------
    >>> cfg = {}
    >>> set_nested_config(cfg, "a.b.c", "123")
    >>> cfg
    {'a': {'b': {'c': 123}}}
    """
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})

    # If value is already a non-string type, set directly
    if not isinstance(value, str):
        d[keys[-1]] = value
        return

    # Convert string value to appropriate type
    if value.isdigit():
        value = int(value)
    elif value == "None":
        value = None
    else:
        try:
            value = float(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
                if isinstance(value, list):
                    value = [
                        float(v) if isinstance(v, (int, float)) and "." in str(v)
                        else int(v) if isinstance(v, (int, float))
                        else v
                        for v in value
                    ]
            except (ValueError, SyntaxError):
                if value.startswith("[") and value.endswith("]"):
                    try:
                        inner = value[1:-1].strip()
                        if inner:
                            items = [item.strip() for item in inner.split(",")]
                            parsed_items = []
                            for item in items:
                                if item == "None":
                                    parsed_items.append(None)
                                elif item.isdigit():
                                    parsed_items.append(int(item))
                                else:
                                    try:
                                        parsed_items.append(float(item))
                                    except ValueError:
                                        parsed_items.append(item)
                            value = parsed_items
                        else:
                            value = []
                    except Exception:
                        pass  # Keep original string value

    d[keys[-1]] = value
