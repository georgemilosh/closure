"""
config.py — Configuration helpers for the closure package.

Provides ``load_paths()`` to resolve data and output directories
from a *paths.yaml* file.
"""

from __future__ import annotations

__all__ = ["load_paths"]

import os

import yaml


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

    # Resolve relative paths against the directory containing paths_file
    base_dir = os.path.dirname(os.path.abspath(paths_file))
    for key in ("work_dir", "data_dir"):
        val = defaults.get(key, "")
        if val and not os.path.isabs(val):
            defaults[key] = os.path.normpath(os.path.join(base_dir, val))

    return defaults
