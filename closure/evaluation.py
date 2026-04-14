"""
evaluation.py — ML evaluation functions for closure.

This module provides functions for evaluating trained models, computing losses,
transforming/unnormalizing predictions, and comparing runs.

Functions accept :class:`~closure.module.ClosureLitModule` and
:class:`~closure.datamodule.ClosureDataModule` objects instead of the
legacy ``Trainer``.
"""

from __future__ import annotations

__all__ = [
    "parse_score",
    "compute_loss",
    "evaluate_loss",
    "evaluate_regression_metrics",
    "transform_features",
    "transform_targets",
    "normalize_input",
    "pred_unnormalize",
    "unnormalize_output",
    "prediction2data",
    "compare_runs",
]

import os
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

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
        except Exception as e:
            print(
                f"Error computing loss with {criterion = }, {ground_truth.shape = }, "
                f"{prediction.shape = }, {type(ground_truth) = }, {type(prediction) = }"
            )
            raise e
    return loss


def evaluate_regression_metrics(
    dataset,
    ground_truth,
    prediction,
    target_channels=None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute per-channel regression metrics in a compact DataFrame.

    Metrics include MSE, RMSE, MAE, R2, Pearson correlation, and
    normalized RMSE (RMSE divided by mean absolute ground truth).

    Parameters
    ----------
    dataset : DataFrameDataset
        Dataset exposing ``request_targets`` and channel metadata.
    ground_truth, prediction : array-like
        Arrays with shape ``(n_samples, n_channels, ...)`` or
        ``(n_samples, n_channels)``.
    target_channels : list[int] or None
        Target channel indices to evaluate. ``None`` means all channels.
    eps : float
        Small stabilizer for divisions.

    Returns
    -------
    pd.DataFrame
        One row per channel with metric columns.
    """
    gt = np.asarray(ground_truth)
    pred = np.asarray(prediction)

    if target_channels is None:
        channel_indices = list(range(len(dataset.request_targets)))
    else:
        channel_indices = list(target_channels)

    rows = []
    for ch in channel_indices:
        y = np.asarray(gt[:, ch]).reshape(-1)
        yhat = np.asarray(pred[:, ch]).reshape(-1)

        err = yhat - y
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))

        denom = float(np.sum((y - np.mean(y)) ** 2))
        if denom <= eps:
            r2 = np.nan
        else:
            r2 = float(1.0 - (np.sum(err**2) / denom))

        std_y = float(np.std(y))
        std_yhat = float(np.std(yhat))
        if std_y <= eps or std_yhat <= eps:
            pearson_r = np.nan
        else:
            pearson_r = float(np.corrcoef(y, yhat)[0, 1])

        mean_abs_y = float(np.mean(np.abs(y)))
        nrmse = float(rmse / (mean_abs_y + eps))

        rows.append(
            {
                "channel_index": ch,
                "channel": dataset.request_targets[ch],
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "pearson_r": pearson_r,
                "mean_abs_ground_truth": mean_abs_y,
                "nrmse": nrmse,
            }
        )

    return pd.DataFrame(rows)


def evaluate_loss(
    dataset,
    ground_truth,
    prediction,
    criterion,
    target_channels=None,
    verbose: bool = True,
) -> dict:
    """Evaluate a loss on total and per-channel level.

    Parameters
    ----------
    dataset : DataFrameDataset
        Dataset providing ``request_targets`` and ``prescaler_targets``.
    ground_truth, prediction : array-like
        Scaled/unscaled target arrays.
    criterion : str
        Loss function name, e.g. ``'MSELoss'``, ``'L1Loss'``, ``'r2'``.
    target_channels : list[int] or None
        Channel indices to evaluate. ``None`` means all channels.
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
        _logger.info("Total loss %s", loss[label])

    if target_channels is None:
        list_of_target_indices = range(len(dataset.prescaler_targets))
    else:
        list_of_target_indices = target_channels

    for channel in list_of_target_indices:
        label = f"{dataset.request_targets[channel]}_{criterion}"
        loss[label] = compute_loss(
            ground_truth[:, channel].flatten(), prediction[:, channel].flatten(), criterion
        )
        if verbose:
            _logger.info(
                "Loss for channel %s: %s, loss = %s",
                channel, dataset.request_targets[channel], loss[label],
            )
    return loss


def transform_features(
    dataset,
    feature_channels=None,
    rescale: bool = True,
    renorm: bool = True,
    verbose: bool = True,
):
    """Un-normalize and un-prescale features from a dataset.

    Parameters
    ----------
    dataset : DataFrameDataset
        Dataset with loaded features and normalization stats.
    feature_channels : list[int] or None
        Feature channel indices. ``None`` means all.
    rescale, renorm : bool
        Whether to undo prescaling / normalization.
    verbose : bool
        Print inverse functions.

    Returns
    -------
    torch.Tensor
        Rescaled features.
    """
    if feature_channels is not None:
        ground_truth = dataset.features[:, feature_channels].squeeze()
    else:
        ground_truth = dataset.features.squeeze()

    pred_shape = [1 for _ in ground_truth.cpu().numpy().shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if feature_channels is None:
        list_of_feature_indices = list(range(len(dataset.request_features)))
    else:
        list_of_feature_indices = feature_channels

    if renorm:
        ground_truth_scaled = (
            ground_truth
            * dataset.features_std[list_of_feature_indices].reshape(pred_shape)
            + dataset.features_mean[list_of_feature_indices].reshape(pred_shape)
        )
    else:
        ground_truth_scaled = ground_truth.clone()

    if rescale:
        for channel in range(len(list_of_feature_indices)):
            if dataset.prescaler_features is not None:
                func = [dataset.prescaler_features[i] for i in list_of_feature_indices][channel]
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
    model,
    dataset,
    target_channels=None,
    rescale: bool = True,
    renorm: bool = True,
    verbose: bool = True,
    reshape: bool = False,
    test_features=None,
):
    """Un-normalize and un-prescale predicted & ground-truth targets.

    Parameters
    ----------
    model : ClosureLitModule
        Lightning module wrapping the network.
    dataset : DataFrameDataset
        Dataset providing features, targets, and normalization stats.
    target_channels : list[int] or None
        Target channel indices. ``None`` means all.
    rescale, renorm : bool
        Whether to undo prescaling / normalization.
    verbose : bool
        Print inverse functions.
    reshape : bool
        Reshape back to spatial dims if True.
    test_features : torch.Tensor, optional
        Custom features for prediction (default: use dataset features).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(ground_truth_scaled, prediction_scaled)``
    """
    if test_features is None:
        features = dataset.features
    else:
        features = test_features

    # Run prediction
    model.eval()
    with torch.no_grad():
        prediction = model(features.to(model.device)).cpu()

    if target_channels is not None:
        ground_truth = dataset.targets[:, target_channels].squeeze()
    else:
        ground_truth = dataset.targets.squeeze()

    pred_shape = [1 for _ in prediction.shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if target_channels is None:
        list_of_target_indices = list(range(len(dataset.prescaler_targets)))
    else:
        list_of_target_indices = target_channels

    if renorm:
        prediction_scaled = (
            prediction * dataset.targets_std[list_of_target_indices].reshape(pred_shape)
            + dataset.targets_mean[list_of_target_indices].reshape(pred_shape)
        )
        ground_truth_scaled = (
            ground_truth * dataset.targets_std[list_of_target_indices].reshape(pred_shape)
            + dataset.targets_mean[list_of_target_indices].reshape(pred_shape)
        )
    else:
        prediction_scaled = prediction.clone()
        ground_truth_scaled = ground_truth.clone()

    if rescale:
        for channel in range(len(list_of_target_indices)):
            func = [dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
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
        if dataset.flatten:
            prediction_scaled = prediction_scaled.reshape((-1,) + dataset.targets_shape[1:])
            ground_truth_scaled = ground_truth_scaled.reshape((-1,) + dataset.targets_shape[1:])

    prediction_scaled = prediction_scaled.cpu().numpy()
    ground_truth_scaled = ground_truth_scaled.cpu().numpy()
    return ground_truth_scaled, prediction_scaled


def normalize_input(data: dict, dataset):
    """Normalize simulation data dict for model inference.

    Parameters
    ----------
    data : dict
        Dictionary containing simulation data arrays.
    dataset : DataFrameDataset
        Dataset providing normalization settings.

    Returns
    -------
    torch.Tensor
        Normalized features ready for inference.
    """
    test_features = []
    for key in dataset.request_features:
        if "_" in key:
            key1, key2 = key.split("_")
            if key1 in data and key2 in data[key1]:
                test_features.append(data[key1][key2])
        else:
            if key in data:
                test_features.append(data[key])

    test_features = np.array(
        test_features, dtype=dataset.features_dtype_numpy
    ).transpose(3, 1, 2, 0)

    if dataset.filter_features is not None:
        test_features = dataset.filter_features(test_features, **dataset.filter_features_kwargs)

    if dataset.flatten:
        test_features = test_features.reshape(-1, test_features.shape[-1])
    else:
        test_features = test_features.transpose(0, 3, 1, 2)

    print(f"{test_features.shape = }")
    dataset._apply_prescaling(
        test_features, dataset.prescaler_features, "features"
    )
    dataset._apply_normalization(test_features, "features")
    test_features = torch.tensor(test_features, dtype=dataset.features_dtype)
    return test_features


def pred_unnormalize(data, test_features, model, dataset, target_channels=None,
                     scaler_targets=None, prescaler_targets=None):
    """Unnormalize model output predictions and inject back into *data*.

    Parameters
    ----------
    data : dict
        Simulation data dictionary (modified in-place).
    test_features : torch.Tensor
        Normalized features used for prediction.
    model : ClosureLitModule
        Lightning module wrapping the network.
    dataset : DataFrameDataset
        Dataset providing normalization settings.
    target_channels : list[int] or None
        Target channel indices. ``None`` means all.
    scaler_targets, prescaler_targets : optional
        Override normalization / prescaling settings.
    """
    model.eval()
    with torch.no_grad():
        prediction = model(test_features.to(model.device)).cpu()
    pred_shape = [1 for _ in prediction.shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if scaler_targets is None:
        scaler_targets = dataset.scaler_targets
    if prescaler_targets is None:
        prescaler_targets = dataset.prescaler_targets

    if target_channels is None:
        list_of_target_indices = list(range(len(dataset.prescaler_targets)))
    else:
        list_of_target_indices = target_channels

    if scaler_targets:
        prediction_scaled = (
            prediction
            * dataset.targets_std[list_of_target_indices].reshape(pred_shape)
            + dataset.targets_mean[list_of_target_indices].reshape(pred_shape)
        )
    else:
        prediction_scaled = prediction

    if prescaler_targets is not None and prescaler_targets is not False:
        for channel in range(len(list_of_target_indices)):
            func = [dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
            if func is None:
                invfunc = lambda a: a
            elif func.__name__ == "log":
                invfunc = torch.exp
            elif func.__name__ == "arcsinh":
                invfunc = torch.sinh
            prediction_scaled[:, channel] = invfunc(prediction_scaled[:, channel])

    if dataset.flatten:
        prediction_scaled = prediction_scaled.reshape(
            dataset.targets_shape[1:] + (-1,)
        ).permute(3, 2, 0, 1)

    print(f"{prediction_scaled.shape = }")
    for i, key in enumerate(dataset.request_targets):
        if "_" in key:
            key1, key2 = key.split("_")
            if key1 in data and key2 in data[key1]:
                data[key1][key2] = prediction_scaled[:, i, ...].numpy().transpose([1, 2, 0])
        else:
            if key in data:
                data[key] = prediction_scaled[:, i, ...].numpy().transpose([1, 2, 0])


# Backward-compatible alias
unnormalize_output = pred_unnormalize


def prediction2data(data: dict, dataset, prediction_scaled):
    """Inject scaled predictions back into *data* dict.

    Parameters
    ----------
    data : dict
        Simulation data dictionary (modified in-place).
    dataset : DataFrameDataset
        Dataset providing metadata (``request_targets``, ``flatten``).
    prediction_scaled : np.ndarray
        Already-scaled predictions.

    Returns
    -------
    dict
        The modified *data* dict.
    """
    for i, key in enumerate(dataset.request_targets):
        if "_" in key:
            key1, key2 = key.split("_")
            if key1 in data and key2 in data[key1]:
                if dataset.flatten:
                    data[key1][key2] = prediction_scaled[..., i].transpose([1, 2, 0])
                else:
                    data[key1][key2] = prediction_scaled[:, i, ...].transpose([1, 2, 0])
        else:
            if key in data:
                if dataset.flatten:
                    data[key] = prediction_scaled[..., i].transpose([1, 2, 0])
                else:
                    data[key] = prediction_scaled[:, i, ...].transpose([1, 2, 0])
    return data


def compare_runs(
    log_dirs: list[str],
    metric_key: str = "val_loss",
) -> pd.DataFrame:
    """Compare metrics across Lightning log directories.

    Reads ``metrics.csv`` files produced by ``CSVLogger`` and returns
    a summary DataFrame.

    Parameters
    ----------
    log_dirs : list[str]
        List of Lightning log directories (each containing ``metrics.csv``).
    metric_key : str
        Column name to extract from the CSV logs.

    Returns
    -------
    pd.DataFrame
        One row per log directory with best metric value and epoch.
    """
    rows = []
    for log_dir in log_dirs:
        csv_path = os.path.join(log_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(log_dir, "version_0", "metrics.csv")
        if not os.path.exists(csv_path):
            rows.append({"log_dir": log_dir, f"best_{metric_key}": None, "best_epoch": None})
            continue

        df = pd.read_csv(csv_path)
        if metric_key not in df.columns:
            rows.append({"log_dir": log_dir, f"best_{metric_key}": None, "best_epoch": None})
            continue

        valid = df.dropna(subset=[metric_key])
        if valid.empty:
            rows.append({"log_dir": log_dir, f"best_{metric_key}": None, "best_epoch": None})
            continue

        best_idx = valid[metric_key].idxmin()
        best_val = valid.loc[best_idx, metric_key]
        best_epoch = int(valid.loc[best_idx, "epoch"]) if "epoch" in valid.columns else best_idx
        rows.append({
            "log_dir": log_dir,
            f"best_{metric_key}": best_val,
            "best_epoch": best_epoch,
        })
    return pd.DataFrame(rows)
