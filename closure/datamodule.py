"""
datamodule.py — ClosureDataModule: PyTorch Lightning data module for closure.

Wraps :class:`~closure.datasets.DataFrameDataset` in Lightning's
``LightningDataModule`` protocol, absorbing channel selection and
subsampling that previously lived in ``ChannelDataLoader``.
"""

from __future__ import annotations

__all__ = ["ClosureDataModule"]

import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import lightning as L

_logger = logging.getLogger("closure.datamodule")

from closure.config import load_paths
from closure.datasets import (
    ChunkOrderedSampler,
    DataFrameDataset,
    FileChunkedSampler,
    LazyNPZDataFrameDataset,
    OnePatchPerFileBatchSampler,
    PreprocessedChunkDataset,
)
from closure.resources import (
    aggregate_gpu_stats,
    cgroup_memory_peak_gb,
    gpu_stats,
    process_tree_ram_gb,
    process_tree_unique_ram_gb,
)


class ClosureDataModule(L.LightningDataModule):
    """Lightning data module for closure training workflows.

    Parameters
    ----------
    data_folder : str
        Root folder containing the simulation data files.
    norm_folder : str
        Folder to save / load normalisation statistics.
    train_samples_file : str
        CSV file listing training sample filenames.
    val_samples_file : str
        CSV file listing validation sample filenames.
    test_samples_file : str or None
        CSV file listing test sample filenames.
    batch_size : int
        Mini-batch size for all dataloaders.
    num_workers : int
        Number of data-loading workers.
    flatten : bool
        If True, flatten spatial dimensions (pixel-wise MLP mode).
    scaler_features : bool or None
        Enable mean/std normalisation for features.
    scaler_targets : bool or None
        Enable mean/std normalisation for targets.
    prescaler_features : list[str | None] or None
        Per-channel prescaler function names (e.g. ``"log"``).
    prescaler_targets : list[str | None] or None
        Per-channel prescaler function names for targets.
    features_dtype : str
        PyTorch dtype name for features (e.g. ``"float32"``).
    targets_dtype : str
        PyTorch dtype name for targets.
    feature_channel_names : list[str] or None
        Subset of feature channels to use (by name).
    target_channel_names : list[str] or None
        Subset of target channels to use (by name).
    subsample_rate : float
        Controls the effective number of training samples per epoch.
        Values below 1.0 select a random subset (undersampling).
        Values above 1.0 repeat samples so each image is visited
        multiple times per epoch (oversampling); useful with
        ``patch_dim`` to extract many random crops per image.
        Default is 1.0 (use all samples exactly once).
    subsample_seed : int or None
        Seed for reproducible subsampling / oversampling.
    patch_dim : list[int] or None
        ``[width, height]`` for random crop patch extraction.
    read_features_targets_kwargs : dict or None
        Extra keyword arguments forwarded to ``read_pic.read_features_targets``.
    filter_features : dict or None
        Spatial filter configuration for features.
    filter_targets : dict or None
        Spatial filter configuration for targets.
    norm_version_dir : str or None
        Optional explicit Lightning ``version_*`` directory to scope
        normalization files for offline evaluation / inference.
    alfven_units : bool
        If True, rescale each sample from code units to Alfvén units
        using the ``.inp`` file auto-detected from its experiment
        subdirectory.  Default ``False``.
    use_readonly : bool
        If True, access ``data_folder`` through the Lustre
        ``/readonly`` mount point.  This avoids aggressive page-cache
        purging on VSC Tier-1 Hortense and can significantly speed up
        repeated HDF5 reads — with zero copy overhead.  The mount is
        read-only, which is fine since data loading never writes.
        Default ``False``.
    loading_mode : str
        ``"eager"`` (default) materializes the full train/val splits in
        RAM via :class:`DataFrameDataset`.  ``"lazy_npz"`` uses
        :class:`LazyNPZDataFrameDataset` and loads one file per
        ``__getitem__`` with a small per-worker LRU cache; required for
        datasets that no longer fit in RAM.  Eager remains the default
        to preserve existing behaviour.
    sample_cache_size : int
        Number of decoded files held per DataLoader worker in lazy
        mode.  Default ``1``.  For ``flatten=True`` (MLP) this should
        match ``chunk_window``.
    chunk_window : int
        Number of files held in flight by
        :class:`FileChunkedSampler` when ``flatten=True`` and
        ``loading_mode="lazy_npz"``.  Larger values interleave more
        files within a batch at the cost of holding more decoded files
        in cache.  Default ``1``.
    persistent_workers : bool or None
        Forwarded to :class:`~torch.utils.data.DataLoader`.  When
        ``None`` (default), set to ``True`` automatically for lazy mode
        with ``num_workers > 0`` (decode cost is paid once per worker
        lifetime) and left ``False`` otherwise.
    prefetch_factor : int or None
        Forwarded to :class:`~torch.utils.data.DataLoader`.  ``None``
        defers to torch's default (2 when ``num_workers > 0``).
    ssd_cache_dir : str or None
        Directory for pre-normalised chunk tensors when
        ``loading_mode="preprocessed"``.  On Hortense, Slurm sets
        ``$TMPDIR`` to the node-local 480 GB NVMe SSD automatically;
        leaving this ``None`` defaults to ``$TMPDIR/closure_preprocessed``
        at ``setup()`` time.  Because ``$TMPDIR`` is wiped between jobs,
        preprocessing runs once per job (not once ever) — still far
        cheaper than lazy NPZ which processes every file every epoch.
        Pass an explicit path to override (must be on a fast local
        filesystem, not Lustre).
    chunk_cache_size : int
        Chunk tensors kept resident per DataLoader worker in
        ``"preprocessed"`` mode.  Default ``1`` (one chunk per worker).
    preprocess_chunk_size_gb : float or None
        RAM budget (GiB) per chunk during the preprocessing pass.
        When ``None`` (default), auto-estimated from
        ``/proc/meminfo`` divided by the number of training GPUs.
    """

    def __init__(
        self,
        data_folder: str,
        norm_folder: str,
        train_samples_file: str,
        val_samples_file: str,
        test_samples_file: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        flatten: bool = True,
        scaler_features: Optional[bool] = None,
        scaler_targets: Optional[bool] = None,
        prescaler_features: Optional[list[str | None]] = None,
        prescaler_targets: Optional[list[str | None]] = None,
        features_dtype: str = "float32",
        targets_dtype: str = "float32",
        features_dtype_numpy: str = "float64",
        targets_dtype_numpy: str = "float64",
        feature_channel_names: Optional[list[str]] = None,
        target_channel_names: Optional[list[str]] = None,
        subsample_rate: float = 1.0,
        subsample_seed: Optional[int] = None,
        patch_dim: Optional[list[int]] = None,
        read_features_targets_kwargs: Optional[dict] = None,
        filter_features: Optional[dict] = None,
        filter_targets: Optional[dict] = None,
        norm_version_dir: Optional[str] = None,
        alfven_units: bool = False,
        use_readonly: bool = False,
        loading_mode: str = "eager",
        sample_cache_size: int = 1,
        chunk_window: int = 1,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        ssd_cache_dir: Optional[str] = None,
        chunk_cache_size: int = 1,
        preprocess_chunk_size_gb: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if loading_mode not in ("eager", "lazy_npz", "preprocessed"):
            raise ValueError(
                f"loading_mode must be 'eager', 'lazy_npz', or 'preprocessed', "
                f"got {loading_mode!r}"
            )
        # ssd_cache_dir is resolved at setup() time (after Slurm sets $TMPDIR).

        # Will be populated in setup()
        self.train_dataset: DataFrameDataset | None = None
        self.val_dataset: DataFrameDataset | None = None
        self.test_dataset: DataFrameDataset | None = None

        # Channel index caches (populated in setup)
        self.feature_channels: list[int] | None = None
        self.target_channels: list[int] | None = None

    # ------------------------------------------------------------------
    # path resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_path(value: str, paths_yaml_key: str) -> str:
        """Resolve a relative path against the corresponding ``paths.yaml`` root.

        * Absolute paths are returned unchanged.
        * Paths starting with ``./`` or ``../`` are treated as explicitly
          relative to the current working directory (resolved to absolute).
        * All other relative paths (bare identifiers such as
          ``ecsim/Harris/Le``) are joined with the directory indicated by
          *paths_yaml_key* (``"data_dir"`` or ``"work_dir"``) from
          ``paths.yaml``.
        """
        p = Path(value)
        if p.is_absolute():
            return str(p)
        if value.startswith(("./", "../")):
            return str(p.resolve())
        root = Path(load_paths().get(paths_yaml_key, "."))
        return str(root / p)

    def _resolve_norm_folder(self, base_norm_folder: str) -> str:
        """Return normalization directory scoped to a run version when available.

        Precedence:
        1) explicit ``hparams.norm_version_dir`` (used by RunLoader)
        2) configured ``base_norm_folder`` when it is explicit (i.e. differs
           from the trainer's ``default_root_dir``, which is what the CLI
           auto-fills when the user provides no value)
        3) active trainer ``log_dir`` (per-version dir used during normal
           training when ``norm_folder`` was not explicitly set)
        4) configured ``base_norm_folder`` fallback
        """
        explicit_version_dir = self.hparams.get("norm_version_dir")
        if explicit_version_dir:
            return str(Path(explicit_version_dir).expanduser().resolve())

        trainer = getattr(self, "trainer", None)
        trainer_log_dir = getattr(trainer, "log_dir", None) if trainer is not None else None
        trainer_root_dir = (
            getattr(trainer, "default_root_dir", None) if trainer is not None else None
        )

        if base_norm_folder:
            base_resolved = str(Path(base_norm_folder).expanduser().resolve())
            if trainer_root_dir:
                root_resolved = str(Path(trainer_root_dir).expanduser().resolve())
                if base_resolved != root_resolved:
                    return base_resolved
            else:
                return base_resolved

        if trainer_log_dir:
            return str(Path(trainer_log_dir).expanduser().resolve())

        return base_norm_folder

    # ------------------------------------------------------------------
    # /readonly Lustre mount
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_readonly_prefix(data_folder: str) -> str:
        """Prepend ``/readonly`` to *data_folder* for Lustre cache retention.

        Paths that already start with ``/readonly`` are returned unchanged.
        A warning is emitted if the resulting path does not exist (e.g.
        running on a non-Hortense system).
        """
        if data_folder.startswith("/readonly"):
            return data_folder
        readonly_path = "/readonly" + data_folder
        if not Path(readonly_path).exists():
            _logger.warning(
                "/readonly mount not available (path %s does not exist). "
                "Falling back to original path %s",
                readonly_path,
                data_folder,
            )
            return data_folder
        _logger.info("Using /readonly mount: %s", readonly_path)
        return readonly_path

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        hp = self.hparams

        self._loading_ram_snapshots_gb: list[float] = []
        self._loading_unique_ram_snapshots_gb: list[float] = []
        self._loading_gpu_util_snapshots_pct: list[float] = []
        self._loading_gpu_mem_snapshots_mb: list[float] = []
        self._loading_ram_peak_gb: float | None = None

        # Resolve relative paths against paths.yaml roots
        data_folder = self._resolve_path(hp.data_folder, "data_dir")

        # /readonly mount: avoids Lustre page-cache purging (zero-cost)
        if hp.use_readonly:
            data_folder = self._apply_readonly_prefix(data_folder)

        norm_folder = self._resolve_path(hp.norm_folder, "work_dir")
        norm_folder = self._resolve_norm_folder(norm_folder)
        train_samples_file = self._resolve_path(hp.train_samples_file, "data_dir")
        val_samples_file = self._resolve_path(hp.val_samples_file, "data_dir")
        test_samples_file = (
            self._resolve_path(hp.test_samples_file, "data_dir")
            if hp.test_samples_file is not None
            else None
        )

        self._log_dataset_plan(
            stage=stage,
            data_folder=data_folder,
            norm_folder=norm_folder,
            train_samples_file=train_samples_file,
            val_samples_file=val_samples_file,
            test_samples_file=test_samples_file,
        )

        # Build common dataset kwargs
        common = dict(
            data_folder=data_folder,
            norm_folder=norm_folder,
            flatten=hp.flatten,
            features_dtype=hp.features_dtype,
            targets_dtype=hp.targets_dtype,
            features_dtype_numpy=hp.features_dtype_numpy,
            targets_dtype_numpy=hp.targets_dtype_numpy,
            scaler_features=hp.scaler_features,
            scaler_targets=hp.scaler_targets,
            prescaler_features=hp.prescaler_features,
            prescaler_targets=hp.prescaler_targets,
            read_features_targets_kwargs=hp.read_features_targets_kwargs,
            filter_features=hp.filter_features,
            filter_targets=hp.filter_targets,
            alfven_units=hp.alfven_units,
        )

        # Select dataset class based on loading mode
        if hp.loading_mode == "lazy_npz":
            dataset_cls = LazyNPZDataFrameDataset
            common["sample_cache_size"] = hp.sample_cache_size
            _logger.info(
                "Lazy NPZ loading enabled (sample_cache_size=%d, chunk_window=%d)",
                hp.sample_cache_size,
                hp.chunk_window,
            )
        elif hp.loading_mode == "preprocessed":
            dataset_cls = PreprocessedChunkDataset
            num_gpus = max(1, getattr(getattr(self, "trainer", None), "num_devices", 1))

            # Resolve ssd_cache_dir: explicit value > $TMPDIR fallback > error.
            ssd_cache_dir = hp.ssd_cache_dir
            if not ssd_cache_dir:
                tmpdir = os.environ.get("TMPDIR", "")
                if tmpdir:
                    ssd_cache_dir = os.path.join(tmpdir, "closure_preprocessed")
                    _logger.info(
                        "ssd_cache_dir not set; using $TMPDIR: %s  "
                        "(SSD is ephemeral — preprocessing runs once per job)",
                        ssd_cache_dir,
                    )
                else:
                    ssd_cache_dir = "/tmp/closure_preprocessed"
                    _logger.warning(
                        "ssd_cache_dir not set and $TMPDIR is unset (interactive session?). "
                        "Falling back to %s — on Hortense GPU nodes /tmp is the local NVMe SSD. "
                        "In a Slurm job, $TMPDIR is set automatically and preferred.",
                        ssd_cache_dir,
                    )

            common["ssd_cache_dir"] = ssd_cache_dir
            common["chunk_cache_size"] = hp.chunk_cache_size
            common["preprocess_chunk_size_gb"] = hp.preprocess_chunk_size_gb
            common["num_gpus"] = num_gpus
            _logger.info(
                "Preprocessed chunked loading enabled (ssd_cache_dir=%s, "
                "chunk_cache_size=%d, preprocess_chunk_size_gb=%s, num_gpus=%d)",
                ssd_cache_dir,
                hp.chunk_cache_size,
                hp.preprocess_chunk_size_gb,
                num_gpus,
            )
        else:
            dataset_cls = DataFrameDataset

        # Build transform for patch extraction (training only).
        # Flattened datasets are pixel-wise vectors, so RandomCrop is invalid there.
        transform = None
        if hp.patch_dim is not None and not hp.flatten:
            transform = {
                "RandomCrop": {"size": hp.patch_dim},
                "apply": ["train"],
            }

        if stage in ("fit", None):
            t0 = time.perf_counter()
            self._log_resource_snapshot("before fit data load")
            self.train_dataset = dataset_cls(
                samples_file=train_samples_file,
                datalabel="train",
                transform=transform,
                **common,
            )
            self._log_dataset_summary(self.train_dataset)
            self._log_resource_snapshot("after train data load")
            self.val_dataset = dataset_cls(
                samples_file=val_samples_file,
                datalabel="val",
                **common,
            )
            self._log_dataset_summary(self.val_dataset)
            self._log_resource_snapshot("after val data load")
            self._data_load_time_s = time.perf_counter() - t0
            _logger.info(
                "Data loading (train+val) took %.2fs",
                self._data_load_time_s,
            )
            # Resolve channel name → index mappings
            self._resolve_channel_indices(self.train_dataset)

        if stage in ("test", "predict", None):
            if test_samples_file is not None:
                self.test_dataset = dataset_cls(
                    samples_file=test_samples_file,
                    datalabel="test",
                    **common,
                )
                self._log_dataset_summary(self.test_dataset)
                self._resolve_channel_indices(self.test_dataset)

    # ------------------------------------------------------------------
    # dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        hp = self.hparams

        if hp.loading_mode == "lazy_npz":
            return self._make_lazy_train_loader()

        if hp.loading_mode == "preprocessed":
            return self._make_preprocessed_train_loader()

        dataset = self._maybe_subsample(self.train_dataset)
        return self._make_loader(dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("No test_samples_file configured.")
        return self._make_loader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()

    # ------------------------------------------------------------------
    # Lightning hook: keep custom samplers in sync with the epoch counter
    # so each epoch reshuffles with a different seed (mirrors what
    # Lightning already does for DistributedSampler).
    # ------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        sampler = getattr(self, "_train_sampler", None)
        if sampler is None or self.trainer is None:
            return
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(self.trainer.current_epoch))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _wrap_channels(self, dataset):
        if self.feature_channels is not None or self.target_channels is not None:
            return _ChannelSubsetDataset(
                dataset, self.feature_channels, self.target_channels
            )
        return dataset

    def _loader_extra_kwargs(self, *, shuffle: bool) -> dict:
        """DataLoader kwargs common to eager and lazy paths."""
        hp = self.hparams
        kwargs: dict = {"pin_memory": True}
        # persistent_workers: keep workers alive across epochs (avoids
        # re-decoding warm caches in lazy mode).  Default ``None`` => True
        # in lazy mode with workers, False otherwise.
        if hp.num_workers > 0:
            if hp.persistent_workers is None:
                kwargs["persistent_workers"] = hp.loading_mode in ("lazy_npz", "preprocessed")
            else:
                kwargs["persistent_workers"] = bool(hp.persistent_workers)
            if hp.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(hp.prefetch_factor)
        return kwargs

    def _make_lazy_train_loader(self) -> DataLoader:
        """Lazy training loader with cache-aware sampler selection.

        - ``flatten=False`` (FCNN/CNN with ``patch_dim``): batches of
          distinct files via :class:`OnePatchPerFileBatchSampler`.
          ``subsample_rate`` (>= 1) becomes the sampler's ``oversample``
          and replaces the eager :meth:`_maybe_subsample` index trick.
        - ``flatten=True`` (MLP): pixel indices grouped per file via
          :class:`FileChunkedSampler` so the per-worker LRU cache stays
          warm.  ``subsample_rate`` is ignored here (one pixel per
          sample already gives ``N*H*W`` samples per epoch); a warning
          is emitted if it deviates from ``1.0``.
        """
        hp = self.hparams
        dataset = self._wrap_channels(self.train_dataset)
        underlying = self.train_dataset
        seed = hp.subsample_seed if hp.subsample_seed is not None else 0
        kwargs = self._loader_extra_kwargs(shuffle=True)

        if not hp.flatten:
            oversample = max(1, int(round(float(hp.subsample_rate))))
            sampler = OnePatchPerFileBatchSampler(
                num_files=underlying.num_files,
                batch_size=hp.batch_size,
                oversample=oversample,
                drop_last=True,
                shuffle=True,
                seed=int(seed),
            )
            self._train_sampler = sampler
            _logger.info(
                "Lazy train loader: OnePatchPerFileBatchSampler "
                "(num_files=%d, batch_size=%d, oversample=%d) -> %d batches/epoch",
                underlying.num_files,
                hp.batch_size,
                oversample,
                len(sampler),
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=hp.num_workers,
                **kwargs,
            )

        if float(hp.subsample_rate) != 1.0:
            _logger.warning(
                "subsample_rate=%s ignored in lazy_npz + flatten=True mode "
                "(one pixel is already one sample)",
                hp.subsample_rate,
            )
        sampler = FileChunkedSampler(
            num_files=underlying.num_files,
            pixels_per_file=underlying._pixels_per_file,
            window=int(hp.chunk_window),
            shuffle=True,
            seed=int(seed),
        )
        self._train_sampler = sampler
        _logger.info(
            "Lazy train loader: FileChunkedSampler "
            "(num_files=%d, pixels_per_file=%d, window=%d)",
            underlying.num_files,
            underlying._pixels_per_file,
            int(hp.chunk_window),
        )
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            sampler=sampler,
            num_workers=hp.num_workers,
            **kwargs,
        )

    def _make_preprocessed_train_loader(self) -> DataLoader:
        """Chunked-SSD training loader with :class:`ChunkOrderedSampler`.

        - ``flatten=False`` (FCNN/CNN): file-level indices, chunk-ordered.
          ``subsample_rate`` (≥ 1) becomes ``oversample`` so each file is
          visited that many times per epoch with a different :class:`_RandomCrop`.
        - ``flatten=True`` (MLP): pixel-level indices, chunk-ordered.
          ``subsample_rate`` is ignored (all pixels emitted once per epoch).
        """
        hp = self.hparams
        dataset = self._wrap_channels(self.train_dataset)
        underlying = self.train_dataset
        seed = hp.subsample_seed if hp.subsample_seed is not None else 0
        kwargs = self._loader_extra_kwargs(shuffle=True)

        if not hp.flatten:
            oversample = max(1, int(round(float(hp.subsample_rate))))
            sampler = ChunkOrderedSampler(
                chunk_sizes=underlying._chunk_sizes,
                pixels_per_file=1,
                oversample=oversample,
                shuffle=True,
                seed=int(seed),
            )
            self._train_sampler = sampler
            _logger.info(
                "Preprocessed train loader: ChunkOrderedSampler FCNN "
                "(chunks=%d, oversample=%d) → %d samples/epoch",
                underlying._num_chunks, oversample, len(sampler),
            )
            return DataLoader(
                dataset,
                batch_size=hp.batch_size,
                sampler=sampler,
                num_workers=hp.num_workers,
                **kwargs,
            )

        if float(hp.subsample_rate) != 1.0:
            _logger.warning(
                "subsample_rate=%s ignored in preprocessed + flatten=True mode "
                "(one pixel is already one sample)",
                hp.subsample_rate,
            )
        sampler = ChunkOrderedSampler(
            chunk_sizes=underlying._chunk_sizes,
            pixels_per_file=underlying._pixels_per_file,
            oversample=1,
            shuffle=True,
            seed=int(seed),
        )
        self._train_sampler = sampler
        _logger.info(
            "Preprocessed train loader: ChunkOrderedSampler MLP "
            "(chunks=%d, pixels/file=%d) → %d samples/epoch",
            underlying._num_chunks, underlying._pixels_per_file, len(sampler),
        )
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            sampler=sampler,
            num_workers=hp.num_workers,
            **kwargs,
        )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        dataset = self._wrap_channels(dataset)
        kwargs = self._loader_extra_kwargs(shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            shuffle=shuffle,
            num_workers=hp.num_workers,
            **kwargs,
        )

    def _maybe_subsample(self, dataset):
        """Return a ``Subset`` with under- or over-sampling applied.

        When ``subsample_rate < 1.0``, a random subset of the dataset is
        selected (undersampling).  When ``subsample_rate > 1.0``, indices
        are repeated so each sample appears multiple times per epoch
        (oversampling).  This is useful with ``patch_dim`` random cropping
        where each access yields a different random patch.
        """
        hp = self.hparams
        if hp.subsample_rate == 1.0:
            return dataset

        n = len(dataset)
        k = max(1, int(n * hp.subsample_rate))
        rng = np.random.RandomState(hp.subsample_seed)

        if hp.subsample_rate < 1.0:
            indices = rng.choice(n, size=k, replace=False).tolist()
        else:
            # Oversampling: cycle indices so each image is visited
            # subsample_rate times per epoch (matching legacy behaviour).
            indices = (rng.permutation(k) % n).tolist()

        return Subset(dataset, indices)

    def _resolve_channel_indices(self, dataset: DataFrameDataset):
        """Convert channel name lists to integer index lists."""
        hp = self.hparams
        if hp.feature_channel_names is not None and self.feature_channels is None:
            self.feature_channels = [
                dataset.request_features.index(ch) for ch in hp.feature_channel_names
            ]
        if hp.target_channel_names is not None and self.target_channels is None:
            self.target_channels = [
                dataset.request_targets.index(ch) for ch in hp.target_channel_names
            ]

    def _log_dataset_plan(
        self,
        stage: str | None,
        data_folder: str,
        norm_folder: str,
        train_samples_file: str,
        val_samples_file: str,
        test_samples_file: str | None,
    ) -> None:
        """Log the resolved dataset inputs before any loading starts."""
        parts = [
            "Resolved dataset inputs",
            f"stage={stage or 'all'}",
            f"data_folder={data_folder}",
            f"norm_folder={norm_folder}",
            f"use_readonly={self.hparams.use_readonly}",
            f"alfven_units={self.hparams.alfven_units}",
            f"train_samples_file={train_samples_file}",
            f"val_samples_file={val_samples_file}",
        ]
        if test_samples_file is not None:
            parts.append(f"test_samples_file={test_samples_file}")
        _logger.info(" | ".join(parts))

    def _log_dataset_summary(self, dataset: DataFrameDataset) -> None:
        """Log what data was loaded and how preprocessing is applied."""
        _logger.info(
            "Dataset ready | split=%s | samples_file=%s | samples=%s | features_shape=%s | targets_shape=%s | flatten=%s",
            dataset.datalabel,
            dataset.samples_file,
            dataset.samples,
            dataset.features_shape,
            dataset.targets_shape,
            dataset.flatten,
        )
        _logger.info(
            "Dataset scaling | split=%s | features=%s | targets=%s | alfven=%s",
            dataset.datalabel,
            self._format_scaling_spec(
                channel_names=dataset.request_features,
                prescalers=dataset.prescaler_features,
                normalized=dataset.scaler_features,
            ),
            self._format_scaling_spec(
                channel_names=dataset.request_targets,
                prescalers=dataset.prescaler_targets,
                normalized=dataset.scaler_targets,
            ),
            self._format_alfven_summary(dataset),
        )

    @staticmethod
    def _format_scaling_spec(
        channel_names: Iterable[str] | None,
        prescalers: Iterable[object] | None,
        normalized: bool | None,
    ) -> str:
        """Return a compact channel->prescaler summary plus normalization flag."""
        names = list(channel_names or [])
        funcs = list(prescalers or [])
        if names and len(funcs) == len(names):
            mapping = ", ".join(
                f"{name}:{getattr(func, '__name__', 'none') if func is not None else 'none'}"
                for name, func in zip(names, funcs)
            )
        elif funcs:
            mapping = ", ".join(
                getattr(func, "__name__", "none") if func is not None else "none"
                for func in funcs
            )
        else:
            mapping = "none"
        return f"prescalers=[{mapping}] normalize={bool(normalized)}"

    @staticmethod
    def _format_alfven_summary(dataset: DataFrameDataset) -> str:
        """Describe Alfvén-unit scaling across all loaded experiments."""
        if not getattr(dataset, "alfven_units", False):
            return "disabled"

        params = list(getattr(dataset, "alfven_params", {}).values())
        if not params:
            return "enabled (scales unavailable)"

        parts = [f"enabled experiments={len(params)}"]
        for key in ("b0x", "nb", "va", "j0", "p0", "e0"):
            values = [float(scale[key]) for scale in params if key in scale]
            if not values:
                continue
            low = min(values)
            high = max(values)
            if abs(high - low) < 1.0e-12:
                parts.append(f"{key}={low:.6g}")
            else:
                parts.append(f"{key}=[{low:.6g}, {high:.6g}]")
        return " ".join(parts)

    def _log_resource_snapshot(self, label: str) -> None:
        """Log RAM/GPU usage snapshot during loading stages."""
        cgroup_ram_gb = process_tree_ram_gb()
        unique_ram_gb = process_tree_unique_ram_gb()
        self._loading_ram_snapshots_gb.append(cgroup_ram_gb)
        self._loading_unique_ram_snapshots_gb.append(unique_ram_gb)

        # Track the cgroup high-water mark (peak since job start).
        peak_gb = cgroup_memory_peak_gb()
        if peak_gb is not None:
            self._loading_ram_peak_gb = peak_gb

        gstats = aggregate_gpu_stats(gpu_stats())
        avg_gpu_util = gstats["avg_gpu_utilization_pct"]
        avg_gpu_mem = gstats["avg_gpu_memory_used_mb"]
        if avg_gpu_util is not None:
            self._loading_gpu_util_snapshots_pct.append(float(avg_gpu_util))
        if avg_gpu_mem is not None:
            self._loading_gpu_mem_snapshots_mb.append(float(avg_gpu_mem))

        parts = [
            f"Loading resource snapshot ({label})",
            f"cgroup_ram_gb={cgroup_ram_gb:.3f}",
            f"unique_ram_gb={unique_ram_gb:.3f}",
        ]
        if peak_gb is not None:
            parts.append(f"cgroup_ram_peak_gb={peak_gb:.3f}")
        if avg_gpu_util is not None:
            parts.append(f"avg_gpu_util={avg_gpu_util:.1f}%")
        if avg_gpu_mem is not None:
            parts.append(f"avg_gpu_mem_mb={avg_gpu_mem:.1f}")
        _logger.info(" | ".join(parts))


class _ChannelSubsetDataset(torch.utils.data.Dataset):
    """Thin wrapper that selects specific feature/target channels."""

    def __init__(self, dataset, feature_channels, target_channels):
        self.dataset = dataset
        self.feature_channels = feature_channels
        self.target_channels = target_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features, targets = self.dataset[idx]
        if self.feature_channels is not None:
            features = features[self.feature_channels]
        if self.target_channels is not None:
            targets = targets[self.target_channels]
        return features, targets
