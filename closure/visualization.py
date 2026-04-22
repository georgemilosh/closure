"""
visualization.py — Plotting helpers for closure.

Functions for visualising model predictions vs ground truth.
"""

from __future__ import annotations

__all__ = ["graph_pred_targets", "plot_pred_targets"]

import os
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    pass

from closure import read_pic as rp


def _sample_cycle_label(sample_filename: str, fallback_index: int) -> str:
    """Extract cycle identifier from a sample filename for stable plot names."""
    basename = os.path.basename(sample_filename)
    cycle_token = basename.rsplit("_", 1)[-1].split(".", 1)[0]
    match = re.search(r"(\d+)$", cycle_token)
    if match:
        return str(int(match.group(1)))
    return str(fallback_index)


def _cycle_range_label(dataset, plot_indices) -> str:
    """Return a compact cycle label for a set of plotted samples."""
    cycle_labels = [
        _sample_cycle_label(dataset.dataframe["filenames"].iloc[i], i)
        for i in plot_indices
    ]
    if not cycle_labels:
        return "none"
    if len(cycle_labels) == 1 or cycle_labels[0] == cycle_labels[-1]:
        return cycle_labels[0]
    return f"{cycle_labels[0]}-{cycle_labels[-1]}"


def graph_pred_targets(
    dataset,
    target_name: str,
    ground_truth_scaled,
    prediction_scaled,
    data_folder: str,
    read_features_targets_kwargs: dict,
    output_dir: str = ".",
    run_name: str = "0",
    reshape: bool = True,
):
    """Generate ground-truth / prediction / error subplots for a target.

    Parameters
    ----------
    dataset : DataFrameDataset
        Dataset with metadata (``request_targets``, ``targets_shape``, etc.).
    target_name : str
        Name of the target variable to visualise.
    ground_truth_scaled, prediction_scaled : array-like
        Scaled arrays from :func:`~closure.evaluation.transform_targets`.
    data_folder : str
        Root data folder for building X/Y grids.
    read_features_targets_kwargs : dict
        Kwargs containing ``choose_x``, ``choose_y``.
    output_dir : str
        Base directory for saving images.
    run_name : str
        Run identifier for output subdirectory.
    reshape : bool
        Reshape to spatial dims (set ``False`` if already reshaped).
    """

    if torch.is_tensor(prediction_scaled):
        prediction_scaled = prediction_scaled.cpu().numpy()
    if torch.is_tensor(ground_truth_scaled):
        ground_truth_scaled = ground_truth_scaled.cpu().numpy()

    channel = dataset.request_targets.index(target_name)
    if reshape:
        prediction_reshaped = prediction_scaled[:, channel].reshape(
            dataset.targets_shape[:-1] + (1,)
        )
        ground_truth_reshaped = ground_truth_scaled[:, channel].reshape(
            dataset.targets_shape[:-1] + (1,)
        )
    else:
        prediction_reshaped = prediction_scaled[..., channel][..., np.newaxis]
        ground_truth_reshaped = ground_truth_scaled[..., channel][..., np.newaxis]

    X, Y = rp.build_XY(
        f"{data_folder}/{dataset.filenames[0].rsplit('/', 1)[0]}/",
        choose_x=read_features_targets_kwargs.get("choose_x"),
        choose_y=read_features_targets_kwargs.get("choose_y"),
    )

    # Scale spatial grids to Alfvén units when training used alfven_units
    if getattr(dataset, "alfven_units", False) and dataset.alfven_params:
        exp_dir = rp._resolve_experiment_dir(
            data_folder, dataset.filenames[0]
        )
        nb = dataset.alfven_params.get(exp_dir, next(iter(dataset.alfven_params.values())))["nb"]
        X = X * np.sqrt(nb)
        Y = Y * np.sqrt(nb)

    img_dir = f"{output_dir}/img/{run_name}"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    _, axs = plt.subplots(3, 3, figsize=(12, 6))

    for i in range(3):
        cycle_label = _sample_cycle_label(dataset.dataframe["filenames"].iloc[i], i)
        error = (ground_truth_reshaped[i, ..., 0] - prediction_reshaped[i, ..., 0]) / (
            ground_truth_reshaped[i, ..., 0].max()
        )
        vmax = ground_truth_reshaped[i, ..., 0].max()
        vmax = [vmax, vmax, 0.5]
        if ground_truth_reshaped[i, ..., 0].min() * ground_truth_reshaped[i, ..., 0].max() > 0:
            vmin = 0
            cmaps = ["plasma", "plasma", "seismic"]
        else:
            vmin = -ground_truth_reshaped[i, ..., 0].max()
            cmaps = ["seismic", "seismic", "seismic"]
        vmin = [vmin, vmin, -0.5]

        for j, (data, label) in enumerate(
            zip(
                [ground_truth_reshaped[i, ..., 0], prediction_reshaped[i, ..., 0], error],
                ["real", "predict", "error"],
            )
        ):
            f, ax = plt.subplots(1, 1, figsize=(6, 3))
            for axes in [ax, axs[i, j]]:
                try:
                    im = axes.pcolormesh(
                        X, Y, data, vmax=vmax[j], vmin=vmin[j], cmap=cmaps[j]
                    )
                except Exception as e:
                    print(f"Error plotting {label} {target_name}, {data.shape = }: {e}")
                axes.set_title(
                    f"{label} {target_name} @ {cycle_label}"
                )
                axes.set_xlabel("X")
                axes.set_ylabel("Y")
                f.colorbar(im, ax=axes)
                f.savefig(
                    f"{img_dir}/{target_name}_cycle{cycle_label}_{label}.png",
                    bbox_inches="tight",
                )
                plt.close(f)

    plt.tight_layout()
    plt.show()


def plot_pred_targets(
    dataset,
    target_name: str,
    prediction,
    ground_truth,
    data_folder: str,
    read_features_targets_kwargs: dict,
    target_channels=None,
    plot_indices=None,
    output_dir: str = ".",
    **kwargs,
):
    """Plot predicted vs ground-truth targets with error panels.

    Each panel is also saved as a separate image file under
    ``<output_dir>/img/``.

    Parameters
    ----------
    dataset : DataFrameDataset
        The test dataset instance.
    target_name : str
        Name of the target variable to visualise.
    prediction : torch.Tensor
        Predicted values (normalised).
    ground_truth : torch.Tensor
        Ground-truth values (normalised).
    data_folder : str
        Root folder of the simulation data.
    read_features_targets_kwargs : dict
        Keyword arguments forwarded to ``read_particles`` for grid
        construction (must contain ``choose_x`` and ``choose_y``).
    target_channels : list[int], optional
        Channel indices used during training. If ``None``, all channels
        are assumed.
    plot_indices : list[int], optional
        Time indices to plot. Defaults to all.
    output_dir : str, optional
        Directory where images are saved (default ``"."``).
    **kwargs
        Forwarded to ``axes.pcolormesh``.
    """
    def _as_numpy(arr):
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        return np.asarray(arr)

    def _robust_absmax(arr, q):
        arr = np.asarray(arr)
        return float(np.quantile(np.abs(arr), q))

    def _compute_main_limits(data, is_signed, q):
        if is_signed:
            vmax_local = _robust_absmax(data, q)
            vmax_local = max(vmax_local, 1e-12)
            return -vmax_local, vmax_local
        vmin_local = 0.0
        vmax_local = float(np.quantile(data, q))
        vmax_local = max(vmax_local, 1e-12)
        return vmin_local, vmax_local

    def _panel_figsize(x_grid, y_grid, base_width=4.5, min_height=3.5, max_height=6.0):
        x_span = float(np.max(x_grid) - np.min(x_grid))
        y_span = float(np.max(y_grid) - np.min(y_grid))
        if x_span <= 0 or y_span <= 0:
            return (base_width, base_width)
        height = base_width * (y_span / x_span)
        height = min(max(height, min_height), max_height)
        return (base_width, height)

    prediction_np = _as_numpy(prediction)
    ground_truth_np = _as_numpy(ground_truth)

    pred_shape = [1 for _ in prediction_np.shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    channel = dataset.request_targets.index(target_name)
    if target_channels is None:
        list_of_target_indices = range(len(dataset.prescaler_targets))
    else:
        list_of_target_indices = target_channels

    func = [dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
    if func is None:
        invfunc = lambda a: a
    elif func.__name__ == "log":
        invfunc = np.exp
    elif func.__name__ == "arcsinh":
        invfunc = np.sinh

    print(f"{invfunc = }")
    X, Y = rp.build_XY(
        f"{data_folder}/{dataset.filenames[0].rsplit('/', 1)[0]}/",
        choose_x=read_features_targets_kwargs.get("choose_x"),
        choose_y=read_features_targets_kwargs.get("choose_y"),
    )

    # Scale spatial grids to Alfvén units when training used alfven_units
    if getattr(dataset, "alfven_units", False) and dataset.alfven_params:
        exp_dir = rp._resolve_experiment_dir(
            data_folder, dataset.filenames[0]
        )
        nb = dataset.alfven_params.get(exp_dir, next(iter(dataset.alfven_params.values())))["nb"]
        X = X * np.sqrt(nb)
        Y = Y * np.sqrt(nb)

    prediction_reshaped = invfunc(
        (
            prediction_np
            * dataset.targets_std[list_of_target_indices].reshape(pred_shape)
            + dataset.targets_mean[list_of_target_indices].reshape(pred_shape)
        )[:, channel]
    ).reshape(dataset.targets_shape[:-1] + (1,))

    ground_truth_reshaped = invfunc(
        (
            ground_truth_np
            * dataset.targets_std[list_of_target_indices].reshape(pred_shape)
            + dataset.targets_mean[list_of_target_indices].reshape(pred_shape)
        )[:, channel]
    ).reshape(dataset.targets_shape[:-1] + (1,))

    if plot_indices is None:
        plot_indices = range(prediction_reshaped.shape[0])

    # Plot controls
    signed_target_names = set(kwargs.pop("signed_target_names", ["Pxy_e", "Pxz_e", "Pyz_e"]))
    robust_quantile = float(kwargs.pop("robust_quantile", 0.995))
    robust_quantile = min(max(robust_quantile, 0.5), 1.0)
    error_mode = kwargs.pop("error_mode", "relative")
    cmap_unsigned = kwargs.pop("cmap_unsigned", "plasma")
    cmap_signed = kwargs.pop("cmap_signed", "seismic")
    cmap_error = kwargs.pop("cmap_error", "seismic")
    error_limit = kwargs.pop("error_limit", None)
    show_figure = bool(kwargs.pop("show_figure", True))
    panel_figsize = kwargs.pop("panel_figsize", None)

    if panel_figsize is None:
        panel_figsize = _panel_figsize(X, Y)

    figsize = kwargs.pop(
        "figsize",
        (3 * panel_figsize[0], len(plot_indices) * panel_figsize[1]),
    )
    fig, axs = plt.subplots(len(plot_indices), 3, figsize=figsize)
    if len(plot_indices) == 1:
        axs = np.asarray([axs])

    img_dir = f"{output_dir}/img"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    target_is_signed = target_name in signed_target_names
    main_cmap = cmap_signed if target_is_signed else cmap_unsigned
    cycle_range_label = _cycle_range_label(dataset, plot_indices)

    for figindex, i in enumerate(plot_indices):
        cycle_label = _sample_cycle_label(dataset.dataframe["filenames"].iloc[i], i)
        gt_i = ground_truth_reshaped[i, ..., 0]
        pred_i = prediction_reshaped[i, ..., 0]

        if error_mode == "absolute":
            error = gt_i - pred_i
        elif error_mode == "symmetric_percent":
            scale = np.quantile(np.abs(gt_i), robust_quantile)
            scale = max(float(scale), 1e-12)
            error = (gt_i - pred_i) / scale
        else:
            # Backward-compatible relative error.
            scale = np.max(np.abs(gt_i))
            scale = max(float(scale), 1e-12)
            error = (gt_i - pred_i) / scale

        vmin_main, vmax_main = _compute_main_limits(gt_i, target_is_signed, robust_quantile)

        if error_limit is None:
            vmax_err = _robust_absmax(error, robust_quantile)
            vmax_err = max(vmax_err, 1e-12)
        else:
            vmax_err = float(abs(error_limit))

        vmin = [vmin_main, vmin_main, -vmax_err]
        vmax = [vmax_main, vmax_main, vmax_err]
        cmaps = [main_cmap, main_cmap, cmap_error]

        for j, (data, label) in enumerate(
            zip(
                [gt_i, pred_i, error],
                ["real", "predict", "error"],
            )
        ):
            f, ax = plt.subplots(1, 1, figsize=panel_figsize)
            for axes in [ax, axs[figindex, j]]:
                im = axes.pcolormesh(
                    X, Y, data, vmax=vmax[j], vmin=vmin[j], cmap=cmaps[j], **kwargs
                )
                axes.set_aspect("equal")
                axes.set_title(
                    f"{label} {target_name} @ {cycle_label}"
                )
                axes.set_xlabel("X")
                axes.set_ylabel("Y")
                axes.figure.colorbar(im, ax=axes)
            f.savefig(
                f"{img_dir}/{target_name}_cycle{cycle_label}_{label}.png",
                bbox_inches="tight",
            )
            plt.close(f)

    plt.tight_layout()
    summary_path = f"{img_dir}/{target_name}_cycles{cycle_range_label}_summary.png"
    fig.savefig(summary_path, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close(fig)
