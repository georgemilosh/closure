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
    resolved_file = paths_file
    if not os.path.isabs(paths_file) and not os.path.exists(paths_file):
        # A relative paths_file is looked up in the CWD, which silently loses
        # the repo configuration whenever a CLI/script/notebook runs from
        # anywhere else (e.g. scripts/ or a diagnostics dir) - downstream that
        # meant e.g. _menura_analysis_dir falling back to whatever
        # menura/analysis it found near the data. Fall back to the repo root
        # (the parent of this package), which also works for editable
        # installs; a site-packages install simply won't find one there and
        # keeps the plain defaults.
        repo_candidate = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), paths_file
        )
        if os.path.exists(repo_candidate):
            resolved_file = repo_candidate
    if os.path.exists(resolved_file):
        with open(resolved_file, "r") as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)

    # Resolve relative paths against the directory containing the file that
    # was actually read. Keep unknown scalar keys (e.g. optional
    # menura_analysis_dir) usable without requiring every path knob to be
    # listed here explicitly.
    base_dir = os.path.dirname(os.path.abspath(resolved_file))
    for key, val in list(defaults.items()):
        if not isinstance(val, str):
            continue
        val = defaults.get(key, "")
        if val and not os.path.isabs(val):
            defaults[key] = os.path.normpath(os.path.join(base_dir, val))

    return defaults
