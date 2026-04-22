"""
run_loader.py — Convenience loader for evaluating trained closure runs.

Provides :class:`RunLoader`, a single object that reconstructs the
``ClosureDataModule``, ``ClosureLitModule``, and test dataset from a
Lightning run directory or config + checkpoint, then exposes simple
methods for computing predictions, metrics, and plots.

Example
-------
>>> from closure import RunLoader
>>> run = RunLoader.from_version_dir("models/Lightning/Harris/Le/lightning_logs/version_0")
>>> run.metrics()
>>> run.plot("Pxx_e")
"""

from __future__ import annotations

__all__ = ["RunLoader"]

import importlib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    pass

from closure.datamodule import ClosureDataModule
from closure.module import ClosureLitModule


_logger = logging.getLogger(__name__)


def _instantiate_network(network_cfg: dict) -> torch.nn.Module:
    """Instantiate a network from a ``class_path`` / ``init_args`` dict."""
    class_path = network_cfg["class_path"]
    init_args = network_cfg.get("init_args", {})
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**init_args)


def _load_module_checkpoint_robust(
    ckpt_path: str | Path,
    network: torch.nn.Module,
    model_cfg: dict,
    device: str,
) -> ClosureLitModule:
    """Load ClosureLitModule checkpoint with Lightning fallback.

    On some HPC module stacks, ``ClosureLitModule.load_from_checkpoint`` can
    fail inside Lightning's legacy migration patch because expected utility
    attributes are missing from shimmed namespaces. In that case we load the
    checkpoint directly via ``torch.load`` and restore the state dict.
    """
    try:
        module = ClosureLitModule.load_from_checkpoint(
            str(ckpt_path),
            network=network,
            map_location=device,
        )
        module.eval()
        return module
    except AttributeError as exc:
        _logger.warning(
            "Lightning load_from_checkpoint failed (%s). Falling back to direct "
            "torch.load state_dict restoration.",
            exc,
        )

    init_kwargs = {k: v for k, v in model_cfg.items() if k != "network"}
    module = ClosureLitModule(network=network, **init_kwargs)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict, strict=True)
    module.eval()
    return module


class RunLoader:
    """One-stop loader for evaluating a trained closure run.

    Holds a :class:`~closure.module.ClosureLitModule` (loaded from
    checkpoint) and a :class:`~closure.datamodule.ClosureDataModule`
    (set up for a given *stage*), so that predictions, metrics, and
    plots are each a single method call.

    Parameters
    ----------
    model : ClosureLitModule
        Lightning module with weights loaded.
    datamodule : ClosureDataModule
        Data module with ``setup()`` already called.
    config : dict
        Raw config dictionary.
    """

    def __init__(
        self,
        model: ClosureLitModule,
        datamodule: ClosureDataModule,
        config: dict,
        version_dir: Path | None = None,
    ):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.version_dir = Path(version_dir) if version_dir is not None else None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_version_dir(
        cls,
        version_dir: str | Path,
        stage: str = "test",
        ckpt: str | Path | None = None,
        device: str | None = None,
        data_overrides: dict[str, Any] | None = None,
    ) -> "RunLoader":
        """Load from a Lightning ``version_*`` directory.

        Parameters
        ----------
        version_dir : str or Path
            Path to a ``lightning_logs/version_X`` directory containing
            ``config.yaml`` and a ``checkpoints/`` subdirectory.
        stage : str
            Lightning stage to set up (``"test"`` or ``"fit"``).
        ckpt : str, Path, or None
            Explicit checkpoint path.  If ``None`` the best checkpoint
            (lowest ``val_loss`` in filename) is selected automatically.
        device : str or None
            Device to load the model on (e.g. ``"cpu"``, ``"cuda"``).
            Defaults to ``"cpu"``.
        data_overrides : dict[str, Any] or None
            Optional overrides merged into ``config['data']`` before
            creating the datamodule (for example ``test_samples_file``).
        """
        version_dir = Path(version_dir)
        config_path = version_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        if ckpt is None:
            ckpt_dir = version_dir / "checkpoints"
            ckpt = cls._find_best_ckpt(ckpt_dir)

        return cls.from_config(
            config_path, ckpt, stage=stage, device=device,
            version_dir=version_dir,
            data_overrides=data_overrides,
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        ckpt_path: str | Path,
        stage: str = "test",
        device: str | None = None,
        version_dir: str | Path | None = None,
        data_overrides: dict[str, Any] | None = None,
    ) -> "RunLoader":
        """Load from an explicit config YAML and checkpoint path.

        Parameters
        ----------
        config_path : str or Path
            Path to the YAML config saved by LightningCLI.
        ckpt_path : str or Path
            Path to the ``.ckpt`` file.
        stage : str
            Lightning stage to set up.
        device : str or None
            Device to load the model on.
        version_dir : str, Path, or None
            Lightning version directory (for history / artifact access).
        data_overrides : dict[str, Any] or None
            Optional overrides merged into ``config['data']`` before
            creating the datamodule (for example ``test_samples_file``).
        """
        device = device or "cpu"

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # --- Rebuild DataModule ---
        data_cfg = dict(cfg.get("data", {}))
        if data_overrides:
            data_cfg.update(data_overrides)
        if version_dir is not None:
            data_cfg["norm_version_dir"] = str(Path(version_dir))
        datamodule = ClosureDataModule(**data_cfg)
        datamodule.setup(stage)

        # --- Rebuild Network + Module ---
        model_cfg = cfg.get("model", {})
        network_cfg = model_cfg.pop("network", None)
        if network_cfg is None:
            raise ValueError("Config is missing model.network section.")
        network = _instantiate_network(network_cfg)

        module = _load_module_checkpoint_robust(
            ckpt_path=ckpt_path,
            network=network,
            model_cfg=model_cfg,
            device=device,
        )

        # Restore network into model_cfg for round-tripping
        model_cfg["network"] = network_cfg

        return cls(
            model=module, datamodule=datamodule, config=cfg,
            version_dir=version_dir,
        )

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------
    @property
    def dataset(self):
        """The test (or current stage) dataset instance."""
        if self.datamodule.test_dataset is not None:
            return self.datamodule.test_dataset
        if self.datamodule.val_dataset is not None:
            return self.datamodule.val_dataset
        if self.datamodule.train_dataset is not None:
            return self.datamodule.train_dataset
        raise RuntimeError("No dataset available — call setup() first.")

    @property
    def target_channels(self):
        """Target channel indices (or ``None`` for all)."""
        return self.datamodule.target_channels

    @property
    def data_folder(self) -> str:
        """Resolved absolute path to the simulation data folder."""
        return ClosureDataModule._resolve_path(
            self.config["data"]["data_folder"], "data_dir"
        )

    @property
    def read_features_targets_kwargs(self) -> dict:
        return self.config["data"].get("read_features_targets_kwargs", {})

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def predict(
        self,
        rescale: bool = False,
        renorm: bool = False,
        reshape: bool = False,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute ground truth and predictions on the test set.

        Parameters
        ----------
        rescale : bool
            Undo prescaling (log / arcsinh).
        renorm : bool
            Undo mean/std normalisation.
        reshape : bool
            Reshape to spatial dims.
        verbose : bool
            Print inverse function info.

        Returns
        -------
        ground_truth : np.ndarray
        prediction : np.ndarray
        """
        from closure.evaluation import transform_targets

        gt, pred = transform_targets(
            self.model,
            self.dataset,
            target_channels=self.target_channels,
            rescale=rescale,
            renorm=renorm,
            verbose=verbose,
            reshape=reshape,
            test_features=self.dataset.features,
        )
        return gt, pred

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def metrics(
        self,
        ground_truth=None,
        prediction=None,
        **kwargs,
    ):
        """Compute per-channel regression metrics.

        If *ground_truth* and *prediction* are not provided they are
        computed via :meth:`predict` (normalised space).

        Returns
        -------
        pd.DataFrame
        """
        from closure.evaluation import evaluate_regression_metrics

        if ground_truth is None or prediction is None:
            ground_truth, prediction = self.predict(**kwargs)

        return evaluate_regression_metrics(
            self.dataset,
            ground_truth,
            prediction,
            target_channels=self.target_channels,
        )

    def loss(
        self,
        criterion: str = "MSELoss",
        ground_truth=None,
        prediction=None,
        **kwargs,
    ) -> dict:
        """Compute total and per-channel loss.

        Returns
        -------
        dict
        """
        from closure.evaluation import evaluate_loss

        if ground_truth is None or prediction is None:
            ground_truth, prediction = self.predict(**kwargs)

        return evaluate_loss(
            self.dataset,
            ground_truth,
            prediction,
            criterion=criterion,
            target_channels=self.target_channels,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(
        self,
        target_name: str,
        ground_truth=None,
        prediction=None,
        output_dir: str = ".",
        **kwargs,
    ):
        """Plot predicted vs ground-truth for a single target channel.

        Parameters
        ----------
        target_name : str
            Name of the target (e.g. ``"Pxx_e"``).
        ground_truth, prediction : array-like or None
            Pre-computed arrays.  If ``None``, computed via :meth:`predict`.
        output_dir : str
            Directory for saved images.
        **kwargs
            Forwarded to :func:`~closure.visualization.plot_pred_targets`
            (e.g. ``figsize``, ``robust_quantile``, ``error_mode``,
            ``signed_target_names``, ``plot_indices``).
        """
        from closure.visualization import plot_pred_targets

        if ground_truth is None or prediction is None:
            ground_truth, prediction = self.predict()

        plot_pred_targets(
            self.dataset,
            target_name,
            prediction,
            ground_truth,
            data_folder=self.data_folder,
            read_features_targets_kwargs=self.read_features_targets_kwargs,
            target_channels=self.target_channels,
            output_dir=output_dir,
            **kwargs,
        )

    def plot_all(
        self,
        ground_truth=None,
        prediction=None,
        output_dir: str = ".",
        **kwargs,
    ):
        """Plot all target channels."""
        if ground_truth is None or prediction is None:
            ground_truth, prediction = self.predict()

        for name in self.dataset.request_targets:
            self.plot(
                name,
                ground_truth=ground_truth,
                prediction=prediction,
                output_dir=output_dir,
                **kwargs,
            )

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------
    def history(self) -> "pd.DataFrame":
        """Load per-epoch training history from ``metrics.csv``.

        Returns a DataFrame with columns ``epoch``, ``train_loss``, and
        ``val_loss`` (one row per epoch).

        Raises
        ------
        FileNotFoundError
            If no ``metrics.csv`` exists in the version directory.
        """
        import pandas as pd

        csv_path = self._metrics_csv_path()
        m = pd.read_csv(csv_path)

        # Lightning's CSVLogger sometimes writes lr on a separate row where
        # ``epoch`` is NaN (logged at the *start* of each epoch).  Back-fill
        # so the LR row inherits the epoch from the next logged row.
        m["epoch"] = m["epoch"].bfill()

        train_epoch = (
            m.dropna(subset=["train_loss"])[["epoch", "train_loss"]]
            .groupby("epoch", as_index=False).last()
        )
        val_epoch = (
            m.dropna(subset=["val_loss"])[["epoch", "val_loss"]]
            .groupby("epoch", as_index=False).last()
        )
        history = train_epoch.merge(val_epoch, on="epoch", how="outer").sort_values("epoch")

        # Attach lr columns if present
        lr_cols = [c for c in m.columns if c.startswith("lr-")]
        for col in lr_cols:
            lr_epoch = (
                m.dropna(subset=[col])[["epoch", col]]
                .groupby("epoch", as_index=False).last()
            )
            history = history.merge(lr_epoch, on="epoch", how="left")

        return history.reset_index(drop=True)

    def plot_history(self, figsize: tuple = (14, 4)):
        """Plot loss and learning-rate curves.

        Parameters
        ----------
        figsize : tuple
            ``(width, height)`` for the figure.
        """
        import matplotlib.pyplot as plt

        h = self.history()
        lr_cols = [c for c in h.columns if c.startswith("lr-")]
        n_panels = 1 + (1 if lr_cols else 0)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]

        axes[0].plot(h["epoch"], h["train_loss"], label="train_loss")
        axes[0].plot(h["epoch"], h["val_loss"], label="val_loss")
        axes[0].set_title("Loss vs epoch")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        if lr_cols:
            for col in lr_cols:
                axes[1].plot(h["epoch"], h[col], label=col)
            axes[1].set_title("Learning rate vs epoch")
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("lr")
            axes[1].grid(alpha=0.3)
            axes[1].legend()

        plt.tight_layout()
        plt.show()

    def best_epoch(self) -> dict:
        """Return a summary dict with best and final epoch statistics."""
        h = self.history()
        val_only = h.dropna(subset=["val_loss"])
        result = {}
        if not val_only.empty:
            best_idx = val_only["val_loss"].idxmin()
            result["best_epoch"] = int(val_only.loc[best_idx, "epoch"])
            result["best_val_loss"] = float(val_only.loc[best_idx, "val_loss"])
        final = h.iloc[-1]
        result["final_epoch"] = int(final["epoch"])
        result["final_train_loss"] = float(final["train_loss"]) if not np.isnan(final["train_loss"]) else None
        result["final_val_loss"] = float(final["val_loss"]) if not np.isnan(final["val_loss"]) else None
        return result

    @classmethod
    def compare_versions(
        cls,
        log_root: str | Path,
        metric_key: str = "val_loss",
    ) -> "pd.DataFrame":
        """Compare runs across all ``version_*`` dirs under *log_root*.

        Parameters
        ----------
        log_root : str or Path
            Directory containing ``version_0/``, ``version_1/``, etc.
        metric_key : str
            Column from ``metrics.csv`` to compare.

        Returns
        -------
        pd.DataFrame
            One row per version with best metric and epoch.
        """
        import pandas as pd

        log_root = Path(log_root)
        versions = sorted(
            [p for p in log_root.glob("version_*") if p.is_dir()],
            key=lambda p: p.name,
        )
        rows = []
        for v in versions:
            csv_path = v / "metrics.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if metric_key not in df.columns:
                continue
            valid = df.dropna(subset=[metric_key])
            if valid.empty:
                continue
            best_idx = valid[metric_key].idxmin()
            rows.append({
                "version": v.name,
                "path": str(v),
                f"best_{metric_key}": float(valid.loc[best_idx, metric_key]),
                "best_epoch": int(valid.loc[best_idx, "epoch"]) if "epoch" in valid.columns else None,
            })
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(f"best_{metric_key}", na_position="last")
        return result

    def _metrics_csv_path(self) -> Path:
        if self.version_dir is not None:
            p = self.version_dir / "metrics.csv"
            if p.exists():
                return p
        raise FileNotFoundError(
            "No metrics.csv found. Set version_dir or load via from_version_dir()."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_best_ckpt(ckpt_dir: Path) -> Path:
        """Pick the checkpoint with the lowest ``val_loss`` from filename."""
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

        best = [c for c in ckpts if "best" in c.stem]
        if best:
            # If multiple "best-*" files, pick by val_loss in name.
            try:
                return min(
                    best,
                    key=lambda p: float(
                        p.stem.split("val_loss")[-1]
                        .replace("=", "")
                        .replace("-", "")
                    ),
                )
            except (ValueError, IndexError):
                return best[0]

        # Fallback: try last.ckpt, then first file.
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return last
        return ckpts[0]

    def __repr__(self) -> str:
        targets = getattr(self.dataset, "request_targets", [])
        vdir = self.version_dir.name if self.version_dir else "?"
        return (
            f"RunLoader({vdir}, targets={targets}, "
            f"n_test_samples={len(self.dataset)}, "
            f"model={self.model.__class__.__name__})"
        )
