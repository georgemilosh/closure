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

from lightning.pytorch.cli import LightningCLI

from closure.module import ClosureLitModule
from closure.datamodule import ClosureDataModule


def main():
    """Launch Lightning CLI."""
    LightningCLI(
        ClosureLitModule,
        ClosureDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
