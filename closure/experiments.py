"""Discovery of experiment (run) folders under a data root.

A PIC "experiment" is a directory holding the per-snapshot field files of a
single simulation run (``*-Fields_*.h5/.npz/.pkl/.vtk`` and friends). Scripts
in this repo take an explicit list of experiment names plus a ``files_path``
root and join the two with :func:`os.path.join`.

When the caller does not specify which experiments to use, we want to fall back
to *every* run folder that actually contains output data under ``files_path``.
:func:`discover_experiments` performs that scan and :func:`resolve_experiments`
wires it into the "use what I gave you, otherwise auto-discover" pattern.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from .read_pic import _collect_experiment_filenames, _resolve_files_path

logger = logging.getLogger(__name__)

__all__ = ["discover_experiments", "resolve_experiments", "has_output_data"]

# Directories that are never themselves experiments and not worth descending
# into when scanning a tree (build artefacts, products, source, VCS, caches).
_SKIP_DIRS = frozenset(
    {
        "build",
        "obj",
        "include",
        "inputs",
        "src",
        "products",
        "plots",
        "logs",
        "__pycache__",
        ".git",
        ".ipynb_checkpoints",
    }
)


def has_output_data(directory: str | os.PathLike[str]) -> bool:
    """Return ``True`` if ``directory`` holds readable PIC field output.

    Uses the same field-file detection as the readers, so a directory counts as
    a run exactly when :func:`closure.read_pic.get_exp_times` could read frames
    from it.
    """
    try:
        return bool(_collect_experiment_filenames(os.fspath(directory)))
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return False


def discover_experiments(
    files_path: str | os.PathLike[str] | None = None,
    *,
    recursive: bool = True,
    max_depth: int | None = None,
) -> list[str]:
    """Find every run folder containing output data under ``files_path``.

    Parameters
    ----------
    files_path
        Root directory containing experiment folders. ``None`` falls back to
        ``data_dir`` from ``paths.yaml`` (same resolution as the readers).
    recursive
        If ``True`` (default) descend into subdirectories looking for runs at
        any depth; a directory that itself holds output data is treated as a
        leaf and not descended into. If ``False`` only the immediate children
        of ``files_path`` are considered.
    max_depth
        Optional cap on recursion depth relative to ``files_path`` (depth 1 ==
        immediate children). Ignored when ``recursive`` is ``False``.

    Returns
    -------
    list[str]
        Experiment names relative to ``files_path`` (POSIX-style separators),
        sorted, each suitable for ``os.path.join(files_path, name)``. The root
        itself is never returned even if it directly holds field files.
    """
    root = _resolve_files_path(files_path)
    if not os.path.isdir(root):
        logger.warning("discover_experiments: %r is not a directory", root)
        return []

    found: list[str] = []

    def _scan(directory: str, depth: int) -> None:
        try:
            entries = sorted(
                (e for e in os.scandir(directory) if e.is_dir(follow_symlinks=False)),
                key=lambda e: e.name,
            )
        except (PermissionError, FileNotFoundError):
            return
        for entry in entries:
            if entry.name in _SKIP_DIRS or entry.name.startswith("."):
                continue
            if has_output_data(entry.path):
                rel = os.path.relpath(entry.path, root)
                found.append(rel.replace(os.sep, "/"))
                # A run folder is a leaf; don't recurse into its products.
                continue
            if recursive and (max_depth is None or depth < max_depth):
                _scan(entry.path, depth + 1)

    _scan(root, 1)
    found.sort()
    logger.info("discover_experiments: found %d run(s) under %s", len(found), root)
    return found


def resolve_experiments(
    experiments: str | Iterable[str] | None,
    files_path: str | os.PathLike[str] | None = None,
    **discover_kwargs,
) -> list[str]:
    """Return an explicit experiment list, auto-discovering when none is given.

    ``experiments`` may be a single name, an iterable of names, or ``None``/empty
    to request discovery of every run folder with output data under
    ``files_path`` (see :func:`discover_experiments`). A non-empty input is
    returned as a list unchanged.
    """
    if experiments is not None and isinstance(experiments, str):
        experiments = [experiments]
    if experiments:
        names = [e for e in experiments if str(e).strip()]
        if names:
            return list(names)
    return discover_experiments(files_path, **discover_kwargs)
