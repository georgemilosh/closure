"""
evaluation.py — ML evaluation functions extracted from utilities.py.

This module provides functions for evaluating trained models, computing losses,
transforming/unnormalizing predictions, and comparing runs.
"""

from __future__ import annotations

__all__ = [
    "parse_score",
    "compute_loss",
    "evaluate_loss",
    "transform_features",
    "transform_targets",
    "normalize_input",
    "pred_unnormalize",
    "unnormalize_output",
    "prediction2data",
    "pred_ground_targets",
    "compare_runs",
    "compare_metrics",
]

import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    pass


def parse_score(score: str):
    """Convert a score name string to the corresponding callable criterion.

    Parameters
    ----------
    score : str
        One of ``'MSE'``, ``'L1Loss'``, or ``'r2'``.

    Returns
    -------
    torch.nn.Module or callable
        The loss / metric object.
    """
    import torchmetrics

    if score in ("MSE", "MSELoss", "L1Loss"):
        name = "MSELoss" if score == "MSE" else score
        return getattr(torch.nn, name)()
    elif score == "r2":
        return torchmetrics.functional.r2_score


def compute_loss(ground_truth, prediction, criterion):
    """Compute a scalar loss between *ground_truth* and *prediction*.

    Parameters
    ----------
    ground_truth : array-like
        Ground-truth values (numpy or torch).
    prediction : array-like
        Predicted values (numpy or torch).
    criterion : str or torch.nn.Module
        Loss name (e.g. ``'MSELoss'``, ``'r2'``) or an instantiated criterion.
    """
    if isinstance(ground_truth, np.ndarray):
        ground_truth = torch.from_numpy(ground_truth)
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)

    if criterion == "r2":
        ss_total = torch.sum((ground_truth - torch.mean(ground_truth)) ** 2)
        ss_residual = torch.sum((ground_truth - prediction) ** 2)
        loss = 1 - (ss_residual / ss_total)
    else:
        try:
            if isinstance(criterion, str):
                loss = getattr(torch.nn, criterion)()(ground_truth, prediction).cpu().numpy()
            elif hasattr(criterion, "__class__") and criterion.__class__.__module__.startswith("torch.nn"):
                loss = criterion(ground_truth, prediction).cpu().numpy()
            else:
                raise ValueError(f"Invalid criterion type: {type(criterion)}")
            loss = getattr(torch.nn, criterion)()(ground_truth, prediction).cpu().numpy()
        except Exception as e:
            print(
                f"Error computing loss with {criterion = }, {ground_truth.shape = }, "
                f"{prediction.shape = }, {type(ground_truth) = }, {type(prediction) = }"
            )
            raise e
    return loss


def evaluate_loss(trainer, ground_truth, prediction, criterion, verbose: bool = True) -> dict:
    """Evaluate a loss on total and per-channel level.

    Parameters
    ----------
    trainer : Trainer
        A Trainer object with loaded run.
    ground_truth, prediction : array-like
        Scaled/unscaled target arrays.
    criterion : str
        Loss function name, e.g. ``'MSELoss'``, ``'L1Loss'``, ``'r2'``.
    verbose : bool
        Print per-channel losses.

    Returns
    -------
    dict
        Mapping ``"total_<criterion>"`` and per-channel losses.
    """
    label = f"total_{criterion}"
    loss = {label: compute_loss(ground_truth.flatten(), prediction.flatten(), criterion)}
    if verbose:
        print(f"Total loss {loss[label]}")

    if trainer.test_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.test_loader.target_channels

    for channel in list_of_target_indices:
        label = f"{trainer.test_dataset.request_targets[channel]}_{criterion}"
        loss[label] = compute_loss(
            ground_truth[:, channel].flatten(), prediction[:, channel].flatten(), criterion
        )
        if verbose:
            print(
                f"Loss for channel {channel}: "
                f"{trainer.test_dataset.request_targets[channel]}, loss = {loss[label]}"
            )
    return loss


def transform_features(trainer, rescale: bool = True, renorm: bool = True, verbose: bool = True):
    """Un-normalize and un-prescale features from the test dataset.

    Parameters
    ----------
    trainer : Trainer
        Trainer with loaded run.
    rescale, renorm : bool
        Whether to undo prescaling / normalization.
    verbose : bool
        Print inverse functions.

    Returns
    -------
    torch.Tensor
        Rescaled features.
    """
    ground_truth = trainer.test_dataset.features[:, trainer.test_loader.feature_channels].squeeze()
    pred_shape = [1 for _ in ground_truth.cpu().numpy().shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if trainer.val_Loader.feature_channels is None:
        list_of_feature_indices = range(
            len(trainer.dataset_kwargs["read_features_targets_kwargs"]["request_features"])
        )
    else:
        list_of_feature_indices = trainer.test_loader.feature_channels

    if renorm:
        ground_truth_scaled = (
            ground_truth
            * trainer.test_dataset.features_std[list_of_feature_indices].reshape(pred_shape)
            + trainer.test_dataset.features_mean[list_of_feature_indices].reshape(pred_shape)
        )
    if rescale:
        for channel, _ in enumerate(trainer.test_dataset.request_features):
            if trainer.test_loader.feature_channels is None:
                list_of_feature_indices = range(
                    len(trainer.dataset_kwargs["read_features_targets_kwargs"]["request_features"])
                )
            else:
                list_of_feature_indices = trainer.test_loader.feature_channels
            if trainer.test_dataset.prescaler_features is not None:
                func = [trainer.test_dataset.prescaler_features[i] for i in list_of_feature_indices][
                    channel
                ]
                if func is None:
                    invfunc = lambda a: a
                elif func.__name__ == "log":
                    invfunc = torch.exp
                elif func.__name__ == "arcsinh":
                    invfunc = torch.sinh
                if verbose:
                    print(f"{invfunc = }")
                ground_truth_scaled[:, channel] = invfunc(ground_truth_scaled[:, channel])
    return ground_truth_scaled


def transform_targets(
    trainer,
    rescale: bool = True,
    renorm: bool = True,
    verbose: bool = True,
    reshape: bool = False,
    test_features=None,
    dataset: str = "test",
):
    """Un-normalize and un-prescale predicted & ground-truth targets.

    Parameters
    ----------
    trainer : Trainer
        Trainer with loaded run.
    rescale, renorm : bool
        Whether to undo prescaling / normalization.
    verbose : bool
        Print inverse functions.
    reshape : bool
        Reshape back to spatial dims if True.
    test_features : torch.Tensor, optional
        Custom features for prediction (default: use dataset features).
    dataset : str
        Which dataset split to use (``'test'``, ``'val'``, ``'train'``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(ground_truth_scaled, prediction_scaled)``
    """
    ds = getattr(trainer, f"{dataset}_dataset")
    loader = getattr(trainer, f"{dataset}_loader")

    if test_features is None:
        prediction = trainer.model.predict(ds.features).cpu()
    else:
        prediction = trainer.model.predict(test_features).cpu()
    ground_truth = ds.targets[:, loader.target_channels].squeeze()
    pred_shape = [1 for _ in prediction.shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if loader.target_channels is None:
        list_of_target_indices = range(len(ds.prescaler_targets))
    else:
        list_of_target_indices = loader.target_channels

    if renorm:
        prediction_scaled = (
            prediction * ds.targets_std[list_of_target_indices].reshape(pred_shape)
            + ds.targets_mean[list_of_target_indices].reshape(pred_shape)
        )
        ground_truth_scaled = (
            ground_truth * ds.targets_std[list_of_target_indices].reshape(pred_shape)
            + ds.targets_mean[list_of_target_indices].reshape(pred_shape)
        )
    if rescale:
        for channel, _ in enumerate(ds.request_targets):
            if loader.target_channels is None:
                list_of_target_indices = range(len(ds.prescaler_targets))
            else:
                list_of_target_indices = loader.target_channels
            func = [ds.prescaler_targets[i] for i in list_of_target_indices][channel]
            if func is None:
                invfunc = lambda a: a
            elif func.__name__ == "log":
                invfunc = torch.exp
            elif func.__name__ == "arcsinh":
                invfunc = torch.sinh
            if verbose:
                print(f"{invfunc = }")
            prediction_scaled[:, channel] = invfunc(prediction_scaled[:, channel])
            ground_truth_scaled[:, channel] = invfunc(ground_truth_scaled[:, channel])

    if reshape:
        if ds.flatten:
            prediction_scaled = prediction_scaled.reshape((-1,) + ds.targets_shape[1:])
            ground_truth_scaled = ground_truth_scaled.reshape((-1,) + ds.targets_shape[1:])

    prediction_scaled = prediction_scaled.cpu().numpy()
    ground_truth_scaled = ground_truth_scaled.cpu().numpy()
    return ground_truth_scaled, prediction_scaled


def normalize_input(data: dict, trainer):
    """Normalize simulation data dict for model inference.

    Parameters
    ----------
    data : dict
        Dictionary containing simulation data arrays.
    trainer : Trainer
        Trainer providing normalization settings.

    Returns
    -------
    torch.Tensor
        Normalized features ready for ``model.predict()``.
    """
    test_features = []
    for key in trainer.test_dataset.request_features:
        if "_" in key:
            key1, key2 = key.split("_")
            if key1 in data and key2 in data[key1]:
                test_features.append(data[key1][key2])
        else:
            if key in data:
                test_features.append(data[key])

    test_features = np.array(
        test_features, dtype=trainer.test_dataset.features_dtype_numpy
    ).transpose(3, 1, 2, 0)

    if trainer.test_dataset.filter_features is not None:
        test_features = trainer.filter_features(test_features, **trainer.filter_features_kwargs)

    if trainer.test_dataset.flatten:
        test_features = test_features.reshape(-1, test_features.shape[-1])
    else:
        test_features = test_features.transpose(0, 3, 1, 2)

    print(f"{test_features.shape = }")
    trainer.test_dataset._apply_prescaling(
        test_features, trainer.test_dataset.prescaler_features, "features"
    )
    trainer.test_dataset._apply_normalization(test_features, "features")
    test_features = torch.tensor(test_features, dtype=trainer.test_dataset.features_dtype)
    return test_features


def pred_unnormalize(data, test_features, trainer, scaler_targets=None, prescaler_targets=None):
    """Unnormalize model output predictions and inject back into *data*.

    Parameters
    ----------
    data : dict
        Simulation data dictionary (modified in-place).
    test_features : torch.Tensor
        Normalized features used for prediction.
    trainer : Trainer
        Trainer providing model and normalization settings.
    scaler_targets, prescaler_targets : optional
        Override normalization / prescaling settings.
    """
    prediction = trainer.model.predict(test_features).cpu()
    pred_shape = [1 for _ in prediction.shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if scaler_targets is None:
        scaler_targets = trainer.test_dataset.scaler_targets
    if prescaler_targets is None:
        prescaler_targets = trainer.test_dataset.prescaler_targets

    if trainer.test_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.test_loader.target_channels

    if scaler_targets:
        prediction_scaled = (
            prediction
            * trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)
            + trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape)
        )
    else:
        prediction_scaled = prediction

    if prescaler_targets is not None and prescaler_targets is not False:
        for channel, _ in enumerate(trainer.test_dataset.request_targets):
            if trainer.test_loader.target_channels is None:
                list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
            else:
                list_of_target_indices = trainer.test_loader.target_channels
            func = [trainer.test_dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
            if func is None:
                invfunc = lambda a: a
            elif func.__name__ == "log":
                invfunc = torch.exp
            elif func.__name__ == "arcsinh":
                invfunc = torch.sinh
            prediction_scaled[:, channel] = invfunc(prediction_scaled[:, channel])

    if trainer.test_dataset.flatten:
        prediction_scaled = prediction_scaled.reshape(
            trainer.test_dataset.targets_shape[1:] + (-1,)
        ).permute(3, 2, 0, 1)

    print(f"{prediction_scaled.shape = }")
    for i, key in enumerate(trainer.test_dataset.request_targets):
        if "_" in key:
            key1, key2 = key.split("_")
            if key1 in data and key2 in data[key1]:
                data[key1][key2] = prediction_scaled[:, i, ...].numpy().transpose([1, 2, 0])
        else:
            if key in data:
                data[key] = prediction_scaled[:, i, ...].numpy().transpose([1, 2, 0])


# Backward-compatible alias
unnormalize_output = pred_unnormalize


def prediction2data(data: dict, trainer, prediction_scaled):
    """Inject scaled predictions back into *data* dict.

    Parameters
    ----------
    data : dict
        Simulation data dictionary (modified in-place).
    trainer : Trainer
        Trainer providing dataset metadata.
    prediction_scaled : np.ndarray
        Already-scaled predictions.

    Returns
    -------
    dict
        The modified *data* dict.
    """
    for i, key in enumerate(trainer.test_dataset.request_targets):
        if "_" in key:
            key1, key2 = key.split("_")
            if key1 in data and key2 in data[key1]:
                if trainer.test_dataset.flatten:
                    data[key1][key2] = prediction_scaled[..., i].transpose([1, 2, 0])
                else:
                    data[key1][key2] = prediction_scaled[:, i, ...].transpose([1, 2, 0])
        else:
            if key in data:
                if trainer.test_dataset.flatten:
                    data[key] = prediction_scaled[..., i].transpose([1, 2, 0])
                else:
                    data[key] = prediction_scaled[:, i, ...].transpose([1, 2, 0])
    return data


def pred_ground_targets(trainer, verbose: bool = True):
    """Return raw predictions and ground-truth targets (deprecated).

    .. deprecated::
        Use :func:`transform_targets` instead.

    Parameters
    ----------
    trainer : Trainer
        Trainer with loaded run.
    verbose : bool
        Print loss values.

    Returns
    -------
    tuple
        ``(prediction, ground_truth, list_of_target_indices)``
    """
    print("The function pred_ground_targets is deprecated. Use transform_targets instead.")
    prediction = trainer.model.predict(trainer.test_dataset.features)
    ground_truth = trainer.test_dataset.targets[:, trainer.test_loader.target_channels].squeeze()
    loss = trainer.model._compute_loss(
        ground_truth.flatten(), prediction.flatten(), trainer.model.criterion
    )
    if verbose:
        print(f"Total loss {loss}")

    if trainer.test_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.test_loader.target_channels

    for channel in list_of_target_indices:
        try:
            loss = trainer.model._compute_loss(
                ground_truth[:, channel].flatten(),
                prediction[:, channel].flatten(),
                trainer.model.criterion,
            )
        except Exception as e:
            print(
                f"{ground_truth.shape = }, {prediction.shape = }, "
                f"{channel = }, {trainer.model.criterion = }"
            )
            raise e
        if verbose:
            print(
                f"Loss for channel {channel}: "
                f"{trainer.test_dataset.request_targets[channel]}, loss = {loss}"
            )
    return prediction, ground_truth, list_of_target_indices


def compare_runs(
    work_dirs: list[str] | None = None,
    runs: list[str] | None = None,
    metric: list[str] | None = None,
    rescale: bool = True,
    renorm: bool = True,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Compare metrics for different runs across work directories.

    Parameters
    ----------
    work_dirs : list[str]
        List of work directories.
    runs : list[str]
        List of run names, one per work_dir.
    metric : list[str], optional
        Additional metrics beyond MSELoss.
    rescale, renorm : bool
        Undo prescaling / normalization.
    verbose : bool
        Print progress.
    **kwargs
        Passed to ``Trainer()``.

    Returns
    -------
    pd.DataFrame
        Comparison results.
    """
    from closure import trainers as tr

    if work_dirs is None:
        work_dirs = ["./"]
    if runs is None:
        runs = ["./0"]

    loss_df = None
    trainer = None
    for i, (work_dir, run) in enumerate(zip(work_dirs, runs)):
        if not os.path.exists(work_dir):
            raise ValueError(f"Work directory '{work_dir}' does not exist.")
        if i == 0 or (i > 0 and os.path.normpath(work_dir) != os.path.normpath(trainer.work_dir)):
            if verbose:
                if i > 0:
                    print(
                        f"Loading trainer from {os.path.normpath(work_dir)} "
                        f"which is different from {os.path.normpath(trainer.work_dir)}"
                    )
                else:
                    print(f"Loading trainer from {work_dir} ")
            trainer = tr.Trainer(work_dir=work_dir, **kwargs)
        if verbose:
            print(f"Loading run {run} ")
        trainer.load_run(run)
        ground_truth_scaled, prediction_scaled = transform_targets(
            trainer, rescale=rescale, renorm=renorm, verbose=verbose
        )
        try:
            score_total = evaluate_loss(
                trainer, ground_truth_scaled, prediction_scaled, "MSELoss", verbose=verbose
            )
        except Exception as e:
            print(f"{ground_truth_scaled.shape = }, {prediction_scaled.shape = }")
            raise e
        if metric is not None:
            for metric_name in metric:
                score_total.update(
                    evaluate_loss(
                        trainer, ground_truth_scaled, prediction_scaled, metric_name, verbose=verbose
                    )
                )

        loss_dict = {"work_dir": work_dir, "exp": work_dir.rsplit("/")[-2], "run": run}
        loss_dict.update(score_total)

        if loss_df is None:
            loss_df = pd.DataFrame(columns=loss_dict.keys())
        loss_df.loc[len(loss_df)] = loss_dict

    return loss_df


def compare_metrics(
    work_dirs: list[str] | None = None,
    runs: list[str] | None = None,
    metric: list[str] | None = None,
) -> pd.DataFrame:
    """Compare metrics for different runs (raw, without un-normalization).

    Parameters
    ----------
    work_dirs : list[str]
        List of work directories.
    runs : list[str]
        List of run names.
    metric : list[str], optional
        Additional metrics.

    Returns
    -------
    pd.DataFrame
        Comparison results.
    """
    from closure import trainers as tr

    if work_dirs is None:
        work_dirs = ["./"]
    if runs is None:
        runs = ["./0"]

    loss_df = None
    for work_dir, run in zip(work_dirs, runs):
        if not os.path.exists(work_dir):
            raise ValueError(f"Work directory '{work_dir}' does not exist.")
        trainer = tr.Trainer(work_dir=work_dir)
        trainer.load_run(run)
        prediction = trainer.model.predict(trainer.test_dataset.features)
        ground_truth = trainer.test_dataset.targets[:, trainer.test_loader.target_channels].squeeze()
        total_loss = (
            trainer.model._compute_loss(
                ground_truth.flatten(), prediction.flatten(), trainer.model.criterion
            )
            .cpu()
            .numpy()
        )
        score = {}
        if metric is not None:
            for metric_name in metric:
                score[f"total_{metric_name}"] = (
                    trainer.model._compute_loss(
                        ground_truth.flatten(), prediction.flatten(), parse_score(metric_name)
                    )
                    .cpu()
                    .numpy()
                )

        if trainer.test_loader.target_channels is None:
            list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
        else:
            list_of_target_indices = trainer.test_loader.target_channels

        loss_dict = {
            "work_dir": work_dir,
            "exp": work_dir.rsplit("/")[-2],
            "run": run,
            "total_loss": total_loss,
        }
        if metric is not None:
            loss_dict.update(score)

        for channel in list_of_target_indices:
            target_loss = trainer.model._compute_loss(
                ground_truth[:, channel].flatten(),
                prediction[:, channel].flatten(),
                trainer.model.criterion,
            )
            loss_dict[trainer.test_dataset.request_targets[channel]] = target_loss.cpu().numpy()
            if metric is not None:
                for metric_name in metric:
                    loss_dict[f"{trainer.test_dataset.request_targets[channel]}_{metric_name}"] = (
                        trainer.model._compute_loss(
                            ground_truth[:, channel].flatten(),
                            prediction[:, channel].flatten(),
                            parse_score(metric_name),
                        )
                        .cpu()
                        .numpy()
                    )

        if loss_df is None:
            loss_df = pd.DataFrame(columns=loss_dict.keys())
        loss_df.loc[len(loss_df)] = loss_dict

    return loss_df
