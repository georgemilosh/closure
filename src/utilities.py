import subprocess
from . import trainers as tr
import pandas as pd
import torch
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
from . import read_pic as rp
import os

def species_to_list(input_list):
    """
    Splits each item in the input_list by '_' if '_' is present in the item.
    
    Args:
        input_list (list): A list of strings.
        
    Returns:
        list: A new list where each item is split by '_' if '_' is present, otherwise the item remains unchanged.

    Example:
        species_to_list(['a', 'b_c', 'd_e_f']) -> ['a', ['b', 'c'], ['d', 'e', 'f']]
    """
    return [item.split('_') if '_' in item else item for item in input_list]


def append_index_to_duplicates(lst):
    """
    This function takes a list as input and returns a new list where each duplicate string element is appended with its 
    index within its group of duplicates. Non-string elements are left unchanged.

    Parameters:
    lst (list): The input list. It can contain elements of any type.

    Returns:
    list: A new list where each duplicate string is appended with its index within its group of duplicates. 
    Non-string elements are left unchanged.
    """
    count_dict = {}
    result = []
    for i, elem in enumerate(lst):
        if isinstance(elem, str):
            if lst.count(elem) > 1:  # Only count duplicates
                if elem in count_dict:
                    count_dict[elem] += 1
                    result.append(f"{elem}{count_dict[elem]}")
                else:
                    count_dict[elem] = 1
                    result.append(f"{elem}{count_dict[elem]}")
            else:
                result.append(elem)
        else:
            result.append(elem)
    return result

def get_duplicate_indices(lst):
    """
    Returns a dictionary containing the indices of duplicate elements in the given list.

    Parameters:
    lst (list): A list of elements.

    Returns:
    dict: A dictionary where the keys are the duplicate elements and the values are lists of their indices.

    Example:
    >>> get_duplicate_indices([1, 2, 3, 2, 4, 1, 5, 4])
    {1: [0, 5], 2: [1, 3], 4: [4, 7]}
    """
    index_dict = {}
    for i, elem in enumerate(lst):
        if elem is not None:
            if elem in index_dict:
                index_dict[elem].append(i)
            else:
                index_dict[elem] = [i]
    return {key: value for key, value in index_dict.items() if len(value) > 1}



def get_git_revision_hash() -> str:
    """
    Returns the hash of the current Git revision. This function assumes that the Git executable is available in the
    system path.

    Returns:
        str: The hash of the current Git revision.
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def parse_score(score):
    """
    This function takes a score name and converts them to class object of this scores"""
    if score in ['MSE', 'L1Loss']:
        return getattr(torch.nn, score)()
    elif score == 'r2':
        return torchmetrics.functional.r2_score

def compare_metrics(work_dirs=['./'], runs=['./0'], metric=None):
    """
    Compare metrics for different runs in the given work directories.

    Args:
        work_dirs (list, optional): List of work directories. Defaults to ['./'].
        runs (list, optional): List of runs. Defaults to ['./0'].
        metric (list, optional): List of metrics to compare. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the comparison results.
    """
    loss_df = None

    for work_dir, run in zip(work_dirs,runs):
        if not os.path.exists(work_dir):
            raise ValueError(f"Work directory '{work_dir}' does not exist.")
        trainer = tr.Trainer(work_dir=work_dir)
        trainer.load_run(run)
        prediction = trainer.model.predict(trainer.test_dataset.features)
        ground_truth = trainer.test_dataset.targets[:,trainer.val_loader.target_channels].squeeze()
        # computing total loss
        total_loss = trainer.model._compute_loss(ground_truth.flatten(),prediction.flatten(),trainer.model.criterion).cpu().numpy()
        score = {}
        if metric is not None:
            for metric_name in metric:
                score[f"total_{metric_name}"] = trainer.model._compute_loss(ground_truth.flatten(),prediction.flatten(),
                                                                 parse_score(metric_name)).cpu().numpy()
        if trainer.train_loader.target_channels is None:
            list_of_target_indices = range(len(trainer.train_dataset.prescaler_targets))
        else:
            list_of_target_indices = trainer.train_loader.target_channels
        loss_dict = {'work_dir': work_dir, 'exp' : work_dir.rsplit('/')[-2],'run': run, 'total_loss': total_loss}
        if metric is not None:
            loss_dict.update(score)
        # computing per channel loss
        for channel in list_of_target_indices:
            target_loss = trainer.model._compute_loss(ground_truth[:, channel].flatten(), prediction[:, channel].flatten(), trainer.model.criterion)
            loss_dict[trainer.train_dataset.request_targets[channel]] = target_loss.cpu().numpy()
            if metric is not None:
                for metric_name in metric:
                    loss_dict[f"{trainer.train_dataset.request_targets[channel]}_{metric_name}"] = \
                        trainer.model._compute_loss(ground_truth[:, channel].flatten(), prediction[:, channel].flatten(),
                                                                 parse_score(metric_name)).cpu().numpy()

        if loss_df is None:
            loss_df = pd.DataFrame(columns=loss_dict.keys())
        loss_df.loc[len(loss_df)] = loss_dict

    return loss_df

def pred_ground_targets(trainer):
    """
    This function takes a trainer object assuming that the run has already been loaded and 
    returns the predicted and ground truth targets.

    Args:
        trainer (Trainer): A Trainer object.

    Returns:
        tuple: A tuple containing the predicted and ground truth targets.
    """
    prediction = trainer.model.predict(trainer.test_dataset.features)
    ground_truth = trainer.test_dataset.targets[:,trainer.val_loader.target_channels].squeeze()
    loss = trainer.model._compute_loss(ground_truth.flatten(),prediction.flatten(),trainer.model.criterion)
    print(f"Total loss {loss}")
    if trainer.train_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.train_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.train_loader.target_channels
    for channel in list_of_target_indices:
        loss = trainer.model._compute_loss(ground_truth[:,channel].flatten(),prediction[:,channel].flatten(),trainer.model.criterion)
        print(f'Loss for channel {channel}:  {trainer.train_dataset.request_targets[channel]}, loss = {loss}')
    return prediction, ground_truth, list_of_target_indices

def plot_pred_targets(trainer, target_name: str, prediction=None, ground_truth=None, 
                      list_of_target_indices=None):
    """
    This function takes a trainer object and a channel index and plots the predicted and ground truth targets along with the errors.
    Each panel is saved as a figure to a file

    Args:
        trainer (Trainer): A Trainer object.
    """

    if prediction is None or ground_truth is None or list_of_target_indices is None:
        prediction, ground_truth, list_of_target_indices = pred_ground_targets(trainer)
    
    pred_shape = [1 for _ in prediction.cpu().numpy().shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    channel = trainer.train_dataset.request_targets.index(target_name)
    if trainer.train_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.train_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.train_loader.target_channels

    func = [trainer.train_dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
    if func == None:
        invfunc = lambda a: a
    elif func.__name__ == 'log':
        invfunc = np.exp
    elif func.__name__ == 'arcsinh':
        invfunc = np.sinh

    print(f"{invfunc = }")
    X, Y = rp.build_XY(f"{trainer.dataset_kwargs['data_folder']}/{trainer.test_dataset.filenames[0].rsplit('/',1)[0]}/",
                        choose_x=trainer.dataset_kwargs['read_features_targets_kwargs']['choose_x'],
                        choose_y = trainer.dataset_kwargs['read_features_targets_kwargs']['choose_y'])
    prediction_reshaped = invfunc((prediction.cpu().numpy()*trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)+
                       trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape))[:,channel]).reshape(trainer.test_dataset.targets_shape[:-1]+(1,))
    ground_truth_reshaped = invfunc((ground_truth.numpy()*trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)+
                         trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape))[:,channel]).reshape(trainer.test_dataset.targets_shape[:-1]+(1,))
    # Create a figure and subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 6))
    if not os.path.exists('img'):
        # Create the directory
        os.makedirs('img')
    # Iterate over the panels
    for i in range(3):
        error = (ground_truth_reshaped[i,...,0] - prediction_reshaped[i,...,0])/(ground_truth_reshaped[i,...,0].max())
        vmax = ground_truth_reshaped[i,...,0].max()
        vmax = [vmax, vmax, .5]
        if ground_truth_reshaped[i,...,0].min()*ground_truth_reshaped[i,...,0].max() > 0:
            vmin = 0
            cmaps = ['plasma', 'plasma', 'seismic']
        else:
            vmin = -ground_truth_reshaped[i,...,0].max()
            cmaps = ['seismic', 'seismic', 'seismic']
        vmin = [vmin, vmin, -.5]
        
        for j, (data,label) in enumerate(zip([ground_truth_reshaped[i,...,0], prediction_reshaped[i,...,0], error],
                                            ['real', 'predict', 'error'])):
            f, ax = plt.subplots(1, 1, figsize=(6, 3))
            for axes in [ax, axs[i,j]]:
                im = axes.pcolormesh(X, Y, data, vmax=vmax[j], vmin=vmin[j], cmap=cmaps[j])
                axes.set_title(f"{label} {target_name} @ {trainer.test_dataset.dataframe['filenames'].iloc[i].rsplit('_')[-1].rsplit('.')[0]}")
                axes.set_xlabel('X')
                axes.set_ylabel('Y')
                f.colorbar(im, ax=axes)
                f.savefig(f'img/{target_name}_time{i}_{label}.png',bbox_inches='tight')
                plt.close(f)
    # Adjust the layout of the subplots
    plt.tight_layout()
    plt.show()