"""
visualization.py — Plotting helpers extracted from utilities.py.

Functions for visualising model predictions vs ground truth.
"""

from __future__ import annotations

__all__ = ["graph_pred_targets", "plot_pred_targets"]

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    pass

from closure import read_pic as rp


def graph_pred_targets(
    trainer,
    target_name: str,
    ground_truth_scaled,
    prediction_scaled,
    reshape: bool = True,
    dataset: str = "test",
):
    """Generate ground-truth / prediction / error subplots for a target.

    Parameters
    ----------
    trainer : Trainer
        Trainer with loaded run.
    target_name : str
        Name of the target variable to visualise.
    ground_truth_scaled, prediction_scaled : array-like
        Scaled arrays from :func:`~closure.evaluation.transform_targets`.
    reshape : bool
        Reshape to spatial dims (set ``False`` if already reshaped).
    dataset : str
        Dataset split (``'test'``, ``'val'``, ``'train'``).
    """
    ds = getattr(trainer, f"{dataset}_dataset")

    if torch.is_tensor(prediction_scaled):
        prediction_scaled = prediction_scaled.cpu().numpy()
    if torch.is_tensor(ground_truth_scaled):
        ground_truth_scaled = ground_truth_scaled.cpu().numpy()

    channel = ds.request_targets.index(target_name)
    if reshape:
        prediction_reshaped = prediction_scaled[:, channel].reshape(
            ds.targets_shape[:-1] + (1,)
        )
        ground_truth_reshaped = ground_truth_scaled[:, channel].reshape(
            ds.targets_shape[:-1] + (1,)
        )
    else:
        prediction_reshaped = prediction_scaled[..., channel][..., np.newaxis]
        ground_truth_reshaped = ground_truth_scaled[..., channel][..., np.newaxis]

    X, Y = rp.build_XY(
        f"{trainer.dataset_kwargs['data_folder']}/{ds.filenames[0].rsplit('/', 1)[0]}/",
        choose_x=trainer.dataset_kwargs["read_features_targets_kwargs"]["choose_x"],
        choose_y=trainer.dataset_kwargs["read_features_targets_kwargs"]["choose_y"],
    )

    img_dir = f"{trainer.work_dir}/img/{trainer.run}"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    _, axs = plt.subplots(3, 3, figsize=(12, 6))

    for i in range(3):
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
                    f"{label} {target_name} @ "
                    f"{ds.dataframe['filenames'].iloc[i].rsplit('_')[-1].rsplit('.')[0]}"
                )
                axes.set_xlabel("X")
                axes.set_ylabel("Y")
                f.colorbar(im, ax=axes)
                f.savefig(
                    f"{img_dir}/{target_name}_time{i}_{label}.png", bbox_inches="tight"
                )
                plt.close(f)

    plt.tight_layout()
    plt.show()


def plot_pred_targets(
    trainer,
    target_name: str,
    prediction=None,
    ground_truth=None,
    list_of_target_indices=None,
    plot_indices=None,
    **kwargs,
):
    """Plot predicted vs ground-truth targets with error panels.

    Each panel is also saved as a separate image file under ``img/``.

    Parameters
    ----------
    trainer : Trainer
        Trainer with loaded run.
    target_name : str
        Name of the target variable to visualise.
    prediction, ground_truth : torch.Tensor, optional
        Predicted / ground-truth values. If ``None``, computed via
        :func:`~closure.evaluation.pred_ground_targets`.
    list_of_target_indices : list, optional
        Channel indices.
    plot_indices : list, optional
        Time indices to plot.
    **kwargs
        Forwarded to ``axes.pcolormesh``.
    """
    from closure.evaluation import pred_ground_targets

    if prediction is None or ground_truth is None or list_of_target_indices is None:
        prediction, ground_truth, list_of_target_indices = pred_ground_targets(trainer)

    pred_shape = [1 for _ in prediction.cpu().numpy().shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    channel = trainer.test_dataset.request_targets.index(target_name)
    if trainer.test_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.test_loader.target_channels

    func = [trainer.test_dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
    if func is None:
        invfunc = lambda a: a
    elif func.__name__ == "log":
        invfunc = np.exp
    elif func.__name__ == "arcsinh":
        invfunc = np.sinh

    print(f"{invfunc = }")
    X, Y = rp.build_XY(
        f"{trainer.dataset_kwargs['data_folder']}"
        f"/{trainer.test_dataset.filenames[0].rsplit('/', 1)[0]}/",
        choose_x=trainer.dataset_kwargs["read_features_targets_kwargs"]["choose_x"],
        choose_y=trainer.dataset_kwargs["read_features_targets_kwargs"]["choose_y"],
    )

    prediction_reshaped = invfunc(
        (
            prediction.cpu().numpy()
            * trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)
            + trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape)
        )[:, channel]
    ).reshape(trainer.test_dataset.targets_shape[:-1] + (1,))

    ground_truth_reshaped = invfunc(
        (
            ground_truth.cpu().numpy()
            * trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)
            + trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape)
        )[:, channel]
    ).reshape(trainer.test_dataset.targets_shape[:-1] + (1,))

    if plot_indices is None:
        plot_indices = range(prediction.shape[-1])
    figsize = kwargs.pop("figsize", (12, 2 * len(plot_indices)))
    fig, axs = plt.subplots(len(plot_indices), 3, figsize=figsize)

    if not os.path.exists("img"):
        os.makedirs("img")

    for figindex, i in enumerate(plot_indices):
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
            f, ax = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1] / 2))
            for axes in [ax, axs[figindex, j]]:
                im = axes.pcolormesh(
                    X, Y, data, vmax=vmax[j], vmin=vmin[j], cmap=cmaps[j], **kwargs
                )
                axes.set_title(
                    f"{label} {target_name} @ "
                    f"{trainer.test_dataset.dataframe['filenames'].iloc[i].rsplit('_')[-1].rsplit('.')[0]}"
                )
                axes.set_xlabel("X")
                axes.set_ylabel("Y")
                f.colorbar(im, ax=axes)
                f.savefig(f"img/{target_name}_time{i}_{label}.png", bbox_inches="tight")
                plt.close(f)

    plt.tight_layout()
    plt.show()
