"""
utlities.py
This module contains utility functions for various tasks such 
as data transformation, loss evaluation, and plotting.
Repo:       closure
Projects:   STRIDE, HELIOSKILL
Author:     George Miloshevich
Date:       2025
License:    MIT License
            
"""

import subprocess
from . import trainers as tr
import pandas as pd
import torch
#import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
from . import read_pic as rp
import re
import os
import pickle
import scipy.ndimage as nd
import ast


def set_nested_config(config, key, value):
    """
    Set a nested configuration value in a dictionary.
    
    This function takes a configuration dictionary, a dot-separated key string, 
    and a value. It sets the value in the dictionary at the location specified 
    by the key string, creating nested dictionaries as needed. The value will 
    be converted to an int, float, or list of ints/floats if possible.
    
    Args:
        config (dict): The configuration dictionary to update.
        key (str): A dot-separated string specifying the nested key.
        value (str): The value to set. It will be converted to an int, float, 
                     or list if possible.
    
    Example:
        config = {}
        set_nested_config(config, 'a.b.c', '123')
        # config is now {'a': {'b': {'c': 123}}}
        
        set_nested_config(config, 'a.b.d', '45.67')
        # config is now {'a': {'b': {'c': 123, 'd': 45.67}}}
        
        set_nested_config(config, 'a.e', '[1, 2, 3]')
        # config is now {'a': {'b': {'c': 123, 'd': 45.67}, 'e': [1, 2, 3]}}
        
        set_nested_config(config, 'a.f', '[1.1, 2.2, 3.3]')
        # config is now {'a': {'b': {'c': 123, 'd': 45.67}, 'e': [1, 2, 3], 'f': [1.1, 2.2, 3.3]}}
        
        set_nested_config(config, 'a.g', '[ReLU,ReLU,ReLU,ReLU]')
        # config is now {'a': {'b': {'c': 123, 'd': 45.67}, 'e': [1, 2, 3], 'f': [1.1, 2.2, 3.3], 'g': ['ReLU', 'ReLU', 'ReLU', 'ReLU']}}
    """

    keys = key.split('.')
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    
    # Convert value to appropriate type
    if value.isdigit():
        value = int(value)
    else:
        try:
            value = float(value)
        except ValueError:
            try:
                # Attempt to parse list using ast.literal_eval first
                value = ast.literal_eval(value)
                if isinstance(value, list):
                    value = [float(v) if isinstance(v, (int, float)) and '.' in str(v) else int(v) if isinstance(v, (int, float)) else v for v in value]
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails, try manual parsing for lists with unquoted strings
                if value.startswith('[') and value.endswith(']'):
                    try:
                        # Remove brackets and split by comma
                        inner = value[1:-1].strip()
                        if inner:
                            items = [item.strip() for item in inner.split(',')]
                            parsed_items = []
                            for item in items:
                                # Try to convert to number, otherwise keep as string
                                if item.isdigit():
                                    parsed_items.append(int(item))
                                else:
                                    try:
                                        parsed_items.append(float(item))
                                    except ValueError:
                                        parsed_items.append(item)
                            value = parsed_items
                        else:
                            value = []
                    except:
                        pass  # Keep original string value if parsing fails
    
    d[keys[-1]] = value


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


def load_and_compute_difference(file_path):
    """
    Load a pickle file containing the training information from the given file path and compute the difference between 'train+val' and 'train' times.
    Parameters:
    file_path (str): The path to the pickle file.
    Returns:
    dict: A dictionary containing the loaded data with the computed difference between 'train+val' and 'train' times stored in 'val' key.
    """
    
    with open(file_path, 'rb') as file:
        loss_dict = pickle.load(file)
    
    loss_dict['time']['val'] = []
    for train, train_val in zip(loss_dict['time']['train'], loss_dict['time']['train+val']):
        loss_dict['time']['val'].append(train_val - train)
    
    return loss_dict

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
    import torchmetrics
    if score in ['MSE', 'L1Loss']:
        return getattr(torch.nn, score)()
    elif score == 'r2':
        return torchmetrics.functional.r2_score

def compare_runs(work_dirs=['./'], runs=['./0'], metric=None, rescale=True, renorm=True, verbose=True, **kwargs):
    """
    Compare metrics for different runs in the given work directories.

    Args:
        work_dirs (list, optional): List of work directories. Defaults to ['./'].
        runs (list, optional): List of runs. Defaults to ['./0'].
        metric (list, optional): List of metrics to compare. Defaults to None.
        **kwargs (dict): Additional keyword arguments to be passed when loading the trainer

    Returns:
        pandas.DataFrame: DataFrame containing the comparison results.
    """
    loss_df = None
    for i, (work_dir, run) in enumerate(zip(work_dirs,runs)):
        if not os.path.exists(work_dir):
            raise ValueError(f"Work directory '{work_dir}' does not exist.")
        if i == 0 or (i > 0 and os.path.normpath(work_dir) != os.path.normpath(trainer.work_dir)): # only request new trainer if work_dir is different from the previous one
            if verbose:
                if i > 0:
                    print(f"Loading trainer from {os.path.normpath(work_dir)} which is different from {os.path.normpath(trainer.work_dir)}")
                else:
                    print(f"Loading trainer from {work_dir} ")
            trainer = tr.Trainer(work_dir=work_dir, **kwargs)
        if verbose:
            print(f"Loading run {run} ")
        trainer.load_run(run)
        ground_truth_scaled, prediction_scaled = transform_targets(trainer, rescale=rescale, renorm=renorm, verbose=verbose)
        score_total = evaluate_loss(trainer, ground_truth_scaled, prediction_scaled, 
                                          'MSELoss', verbose=verbose)
        if metric is not None:
            for metric_name in metric:
                score_total.update(evaluate_loss(trainer, ground_truth_scaled, prediction_scaled, 
                                                   metric_name, verbose=verbose))
                
        loss_dict = {'work_dir': work_dir, 'exp' : work_dir.rsplit('/')[-2],'run': run}
        loss_dict.update(score_total)
       
        if loss_df is None:
            loss_df = pd.DataFrame(columns=loss_dict.keys())
        loss_df.loc[len(loss_df)] = loss_dict

    return loss_df

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
        ground_truth = trainer.test_dataset.targets[:,trainer.test_loader.target_channels].squeeze()
        # computing total loss
        total_loss = trainer.model._compute_loss(ground_truth.flatten(),prediction.flatten(),trainer.model.criterion).cpu().numpy()
        score = {}
        if metric is not None:
            for metric_name in metric:
                score[f"total_{metric_name}"] = trainer.model._compute_loss(ground_truth.flatten(),prediction.flatten(),
                                                                 parse_score(metric_name)).cpu().numpy()
        if trainer.test_loader.target_channels is None:
            list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
        else:
            list_of_target_indices = trainer.test_loader.target_channels
        loss_dict = {'work_dir': work_dir, 'exp' : work_dir.rsplit('/')[-2],'run': run, 'total_loss': total_loss}
        if metric is not None:
            loss_dict.update(score)
        # computing per channel loss
        for channel in list_of_target_indices:
            target_loss = trainer.model._compute_loss(ground_truth[:, channel].flatten(), prediction[:, channel].flatten(), trainer.model.criterion)
            loss_dict[trainer.test_dataset.request_targets[channel]] = target_loss.cpu().numpy()
            if metric is not None:
                for metric_name in metric:
                    loss_dict[f"{trainer.test_dataset.request_targets[channel]}_{metric_name}"] = \
                        trainer.model._compute_loss(ground_truth[:, channel].flatten(), prediction[:, channel].flatten(),
                                                                 parse_score(metric_name)).cpu().numpy()

        if loss_df is None:
            loss_df = pd.DataFrame(columns=loss_dict.keys())
        loss_df.loc[len(loss_df)] = loss_dict

    return loss_df

def conserved_quantities(folder, verbose=True):
    """
    Reads ConservedQuantities.txt generated by ECsim containing conserved quantities from a specified folder,
    and returns the data as a pandas DataFrame.
    Args:
        folder (str): The path to the folder containing the CSV file.
        verbose (bool, optional): If True, prints the list of variable names. Defaults to True.
    Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file, with columns named appropriately.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an error reading the file.
    Usage:
        >>> 
            conserved_quantities("path/to/folder")
            import matplotlib.pyplot as plt

            # Create subplots with 3 rows and 3 columns, and adjust the figsize parameter
            fig, axs = plt.subplots(3, 3, figsize=(12, 6))

            # Iterate over the axes and plot the data
            for i, ax in enumerate(axs.flatten()):
                data.iloc[:, i].plot(ax=ax)
                ax.set_ylabel(f'{variables[i+1]}')
                ax.set_xlabel('cycles')

            # Adjust the layout of the subplots
            plt.tight_layout()

            # Show the plot
            plt.show()
        >>>

    """
    file_path = f"{folder}/ConservedQuantities.txt"

    # Define column names based on the provided structure
    column_names = [
        "Cycle",
        "Total internal energy",
        "Variation of total internal energy",
        "Electric energy",
        "Local magnetic energy",
        "Kinetic energy (currently in the domain)",
        "Momentum",
        "Total magnetic energy",
        "Internal magnetic energy",
        "Kinetic energy removed",
        "Electric energy removed",
        "Number of particles of species 0",
        "Total charge of species 0",
        "Kinetic energy of species 0",
        "Number of particles of species 1",
        "Total charge of species 1",
        "Kinetic energy of species 1"
    ]

    # Read the data from the file
    data = pd.read_csv(file_path, sep=r"\s+", header=None, names=column_names)
    data = data.drop(columns=['Cycle'])
    
    if verbose:
        print("variables ", column_names[1:])
    
    return data, column_names

def transform_features(trainer, rescale=True, renorm=True, verbose=True):
    """
    Transforms the features based on the trainer's configuration.
    Args:
        trainer: The trainer object containing the model, test dataset, and train dataset.
        rescale (bool): Whether to rescale the features.
        renorm (bool): Whether to renormalize the features.
        verbose (bool): Whether to print the loss values.
    Returns:
        features_scaled: The scaled features.
    """
    
    ground_truth = trainer.test_dataset.features[:,trainer.test_loader.feature_channels].squeeze()
    pred_shape = [1 for _ in ground_truth.cpu().numpy().shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if trainer.val_Loader.feature_channels is None: 
        list_of_feature_indices = range(len(trainer.dataset_kwargs['read_features_targets_kwargs']['request_features']))
    else:
        list_of_feature_indices = trainer.test_loader.feature_channels

    if renorm:
        ground_truth_scaled = (ground_truth*trainer.test_dataset.features_std[list_of_feature_indices].reshape(pred_shape)+
                                trainer.test_dataset.features_mean[list_of_feature_indices].reshape(pred_shape))
    if rescale:
        for channel, _ in enumerate(trainer.test_dataset.request_features):
            if trainer.test_loader.feature_channels is None:
                list_of_feature_indices = range(len(trainer.dataset_kwargs['read_features_targets_kwargs']['request_features']))
            else:
                list_of_feature_indices = trainer.test_loader.feature_channels
            if trainer.test_dataset.prescaler_features is not None:
                func = [trainer.test_dataset.prescaler_features[i] for i in list_of_feature_indices][channel]
                if func == None:
                    invfunc = lambda a: a
                elif func.__name__ == 'log':
                    invfunc = torch.exp
                elif func.__name__ == 'arcsinh':
                    invfunc = torch.sinh
                if verbose:
                    print(f"{invfunc = }")
                ground_truth_scaled[:,channel] = invfunc(ground_truth_scaled[:,channel])
    return ground_truth_scaled

def transform_targets(trainer, rescale=True, renorm=True, verbose=True):
    """
    Transforms the predicted and ground truth targets based on the trainer's configuration.
    Args:
        trainer: The trainer object containing the model, test dataset, and train dataset.
        rescale (bool): Whether to rescale the targets.
        renorm (bool): Whether to renormalize the targets.
        verbose (bool): Whether to print the loss values.
    Returns:
        prediction_scaled: The scaled predicted targets.
        ground_truth_scaled: The scaled ground truth targets.
    """

    prediction = trainer.model.predict(trainer.test_dataset.features).cpu()
    ground_truth = trainer.test_dataset.targets[:,trainer.test_loader.target_channels].squeeze()
    pred_shape = [1 for _ in prediction.shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if trainer.test_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.test_loader.target_channels

    if renorm:
        prediction_scaled = (prediction*trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)+
                            trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape))
        ground_truth_scaled = (ground_truth*trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)+
                                trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape))
    if rescale:
        for channel, _ in enumerate(trainer.test_dataset.request_targets):
            if trainer.test_loader.target_channels is None:
                list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
            else:
                list_of_target_indices = trainer.test_loader.target_channels

            func = [trainer.test_dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
            if func == None:
                invfunc = lambda a: a
            elif func.__name__ == 'log':
                invfunc = torch.exp
            elif func.__name__ == 'arcsinh':
                invfunc = torch.sinh
            if verbose:
                print(f"{invfunc = }")
            prediction_scaled[:,channel] = invfunc(prediction_scaled[:,channel])
            ground_truth_scaled[:,channel] = invfunc(ground_truth_scaled[:,channel])
    return ground_truth_scaled, prediction_scaled 

def compute_loss(ground_truth, prediction, criterion):
    if criterion == 'r2':
        #loss = (1- torch.nn.MSELoss()(ground_truth,prediction)/torch.var(ground_truth)).cpu().numpy()
        ss_total = torch.sum((ground_truth - torch.mean(ground_truth)) ** 2)
        ss_residual = torch.sum((ground_truth - prediction) ** 2)
        loss = 1 - (ss_residual / ss_total)
    else:
        loss = getattr(torch.nn, criterion)()(ground_truth,prediction).cpu().numpy()
    return loss

def evaluate_loss(trainer, ground_truth, prediction, criterion, verbose=True):
    """
    This function takes a trainer object assuming that the run has already been loaded and 
    returns the loss

    Args:
        trainer (Trainer): A Trainer object.
        ground_truth (torch.Tensor): The ground truth targets.
        prediction (torch.Tensor): The predicted targets.
        criterion (str): The loss function to use, e.g. 'MSELoss', 'L1Loss', 'r2'.
    Returns:
        tuple: A tuple containing the predicted and ground truth targets.
    """
    label = f'total_{criterion}'
    loss = {label : compute_loss(ground_truth.flatten(),prediction.flatten(),criterion)}
    if verbose:
        print(f"Total loss {loss[label]}")
    if trainer.test_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.test_loader.target_channels
    for channel in list_of_target_indices:
        label = f'{trainer.test_dataset.request_targets[channel]}_{criterion}'
        loss[label] = compute_loss(ground_truth[:,channel].flatten(),prediction[:,channel].flatten(),criterion)
        if verbose:
            print(f'Loss for channel {channel}:  {trainer.test_dataset.request_targets[channel]}, loss = {loss[label]}')
    return loss

def graph_pred_targets(trainer, target_name: str, ground_truth_scaled, prediction_scaled):
    """
    Generate and display a grid of subplots showing the ground truth, predictions, and error for a specific target variable.
    Parameters:
    - trainer: The trainer object containing the datasets and other necessary information.
    - target_name: The name of the target variable to visualize.
    - ground_truth_scaled: The scaled ground truth values for the target variable.
    - prediction_scaled: The scaled predicted values for the target variable.
    Returns:
    None
    """
    print("The function graph_pred_targets is deprecated. Use pred_ground_targets instead.")
    channel = trainer.test_dataset.request_targets.index(target_name)
    prediction_reshaped = prediction_scaled[:,channel].reshape(trainer.test_dataset.targets_shape[:-1]+(1,)).cpu().numpy()
    ground_truth_reshaped = ground_truth_scaled[:,channel].reshape(trainer.test_dataset.targets_shape[:-1]+(1,)).cpu().numpy()

    X, Y = rp.build_XY(f"{trainer.dataset_kwargs['data_folder']}/{trainer.test_dataset.filenames[0].rsplit('/',1)[0]}/",
                        choose_x=trainer.dataset_kwargs['read_features_targets_kwargs']['choose_x'],
                        choose_y = trainer.dataset_kwargs['read_features_targets_kwargs']['choose_y'])
    import os
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

def pred_ground_targets(trainer, verbose=True):
    """
    This function takes a trainer object assuming that the run has already been loaded and 
    returns the predicted and ground truth targets.

    Args:
        trainer (Trainer): A Trainer object.

    Returns:
        tuple: A tuple containing the predicted and ground truth targets.
    """
    print("The function pred_ground_targets is deprecated. Use transform_targets instead.")
    prediction = trainer.model.predict(trainer.test_dataset.features)
    ground_truth = trainer.test_dataset.targets[:,trainer.test_loader.target_channels].squeeze()
    loss = trainer.model._compute_loss(ground_truth.flatten(),prediction.flatten(),trainer.model.criterion)
    if verbose:
        print(f"Total loss {loss}")
    if trainer.test_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.test_loader.target_channels
    for channel in list_of_target_indices:
        try:
            loss = trainer.model._compute_loss(ground_truth[:,channel].flatten(),prediction[:,channel].flatten(),trainer.model.criterion)
        except Exception as e:
            print(f"{ground_truth.shape = }, {prediction.shape = }, {channel = }, {trainer.model.criterion = }")
            raise e
        if verbose:
            print(f'Loss for channel {channel}:  {trainer.test_dataset.request_targets[channel]}, loss = {loss}')
    return prediction, ground_truth, list_of_target_indices

def plot_pred_targets(trainer, target_name: str, prediction=None, ground_truth=None, 
                      list_of_target_indices=None, plot_indices=None, **kwargs):
    """
    This function takes a trainer object and a channel index and plots the predicted and ground truth targets along with the errors.
    Each panel is saved as a figure to a file

    Args:
        trainer (Trainer): A Trainer object.
        target_name (str): The name of the target variable to visualize.
        prediction (torch.Tensor): The predicted values for the target variable.
        ground_truth (torch.Tensor): The ground truth values for the target variable.
        list_of_target_indices (list): A list of target indices.
        plot_indices (list): A list of indices to plot, basically which times to plot.
        **kwargs: Additional keyword arguments to be passed to the plotting functions: axes.pcolormesh
    """
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
    ground_truth_reshaped = invfunc((ground_truth.cpu().numpy()*trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)+
                         trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape))[:,channel]).reshape(trainer.test_dataset.targets_shape[:-1]+(1,))
    # Create a figure and subplots
    if plot_indices is None:
        plot_indices = range(prediction.shape[-1])
    figsize = kwargs.pop('figsize', (12, 2*len(plot_indices)))
    fig, axs = plt.subplots(len(plot_indices), 3, figsize=figsize)
    if not os.path.exists('img'):
        # Create the directory
        os.makedirs('img')
    # Iterate over the panels
    for figindex, i in enumerate(plot_indices):
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
            f, ax = plt.subplots(1, 1, figsize=(figsize[0]/2, figsize[1]/2)) # we are also saving the individual plots
            for axes in [ax, axs[figindex,j]]:
                im = axes.pcolormesh(X, Y, data, vmax=vmax[j], vmin=vmin[j], cmap=cmaps[j], **kwargs)
                axes.set_title(f"{label} {target_name} @ {trainer.test_dataset.dataframe['filenames'].iloc[i].rsplit('_')[-1].rsplit('.')[0]}")
                axes.set_xlabel('X')
                axes.set_ylabel('Y')
                f.colorbar(im, ax=axes)
                f.savefig(f'img/{target_name}_time{i}_{label}.png',bbox_inches='tight')
                plt.close(f)
    # Adjust the layout of the subplots
    plt.tight_layout()
    plt.show()

def normalize_input(data, trainer, prescaler_features=None, scaler_features=None):
    """
    Normalize the input data based on the trainer's dataset settings.

    Parameters:
    data (dict): Dictionary containing the simulation data.
    trainer (Trainer): Trainer object containing the model and normalization settings.

    Returns:
    torch.Tensor: Normalized test features ready for model prediction.
    """
    test_features = []
    for key in trainer.test_dataset.request_features:
        if '_' in key:
            key1, key2 = key.split('_')
            if key1 in data and key2 in data[key1]:
                test_features.append(data[key1][key2])
        else:
            if key in data:
                test_features.append(data[key])
    if scaler_features is None:
        scaler_features = trainer.test_dataset.scaler_features
    if prescaler_features is None:
        prescaler_features = trainer.test_dataset.prescaler_features
    # Concatenate the selected data into one array
    test_features = np.array(test_features).transpose([3, 0, 1, 2])
    print("original std = ", np.std(test_features, axis=(0, 2, 3)))
    if prescaler_features is not None and prescaler_features is not False and trainer.test_dataset.prescaler_features is not None:
        for channel in range(trainer.test_dataset.features.shape[1]):
            if trainer.test_dataset.prescaler_features[channel] is not None:
                test_features[:,channel,...] =  trainer.test_dataset.prescaler_features[channel](test_features[:,channel,...])
    if scaler_features:
        for channel in range(trainer.test_dataset.features.shape[1]):
            try:
                test_features[:,channel,...] -= trainer.test_dataset.features_mean[channel]
            except Exception as e:
                print(f"{test_features.shape = }, {trainer.test_dataset.features_mean = }")
                raise e
            test_features[:,channel,...] /= trainer.test_dataset.features_std[channel]
    print("after normalization std = ", np.std(test_features, axis=(0, 2, 3)))
    test_features = torch.tensor(test_features, dtype=trainer.test_dataset.feature_dtype)
    return test_features

def unnormalize_output(data, test_features, trainer, scaler_targets=None, prescaler_targets=None):
    """
    Unnormalize the output predictions based on the trainer's dataset settings.

    Parameters:
    data (dict): Dictionary containing the simulation data.
    test_features (torch.Tensor): Normalized test features used for model prediction.
    trainer (Trainer): Trainer object containing the model and normalization settings.

    Returns:
    None: The function updates the `data` dictionary with the unnormalized predictions.
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
        prediction_scaled = (prediction*trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)+
                            trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape))
    else:
        prediction_scaled = prediction
    if prescaler_targets is not None and prescaler_targets is not False:
        for channel, _ in enumerate(trainer.test_dataset.request_targets):
            if trainer.test_loader.target_channels is None:
                list_of_target_indices = range(len(trainer.test_dataset.prescaler_targets))
            else:
                list_of_target_indices = trainer.test_loader.target_channels

            func = [trainer.test_dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
            if func == None:
                invfunc = lambda a: a
            elif func.__name__ == 'log':
                invfunc = torch.exp
            elif func.__name__ == 'arcsinh':
                invfunc = torch.sinh
            prediction_scaled[:,channel] = invfunc(prediction_scaled[:,channel])

    for i, key in enumerate(trainer.test_dataset.request_targets):
        if '_' in key:
            key1, key2 = key.split('_')
            if key1 in data and key2 in data[key1]:
                data[key1][key2] = prediction_scaled[:, i, ...].numpy().transpose([1, 2, 0])
        else:
            if key in data:
                data[key] = prediction_scaled[:, i, ...].numpy().transpose([1, 2, 0])
                


# The scripts below are adapted from G. Arrò

def scalar_spectrum_2D(field, X, Y):
    """
    Author: Peppe Arrò
    This script calculates the 1D power spectrum for scalar functions
    """
    Lx = X[-1,0]
    Ly = Y[0,-1]
    x = X[:,0]
    y = Y[0,:]
    t = np.arange(field.shape[-1])
    nxc=len(X)-1
    nyc=len(Y)-1
    print(f"{field.shape = }, {nxc = }, {nyc = }, {x.shape = }, {y.shape = }, {t.shape = }, {x[:2] = }, {y[:2] = }")
    # Repeated boundaries must be excluded according to the definition of the FFT.
    field_ft=np.fft.rfft2(field[0:-1,0:-1,:],axes=(0,1))
    print(f"{field_ft.shape = }")
    # 2D power spectrum.
    spec_2D=(abs(field_ft)**2)/((nxc*nyc)**2)
    spec_2D[:,1:-1,:]*=2 # Some modes are doubled to take into account the redundant ones removed by numpy's rfft.
    kx=np.fft.fftfreq(nxc-1,x[1])*2*np.pi
    ky=np.fft.rfftfreq(nyc-1,y[1])*2*np.pi
    print(f"{len(kx) = }, {len(ky) = }")

    # The 1D magnetic field energy spectrum is calculated.
    spec_1D=np.zeros((nxc//2+1,len(t)))

    for iy in range(len(ky)):
        for ix in range(len(kx)):
            try:
                index=round( np.sqrt( (Lx*kx[ix]/(2*np.pi))**2+(Ly*ky[iy]/(2*np.pi))**2 ) )
                if index<=(nxc//2):
                    spec_1D[index,:]+=spec_2D[ix,iy,:]
            except Exception as e:
                print(f"{index = }, {ix = }, {iy = }, {spec_2D.shape = }, {spec_1D.shape = }")
                raise e

    return ky,spec_1D[:-1]

def vector_spectrum_2D(field_x,field_y,field_z, X, Y):
    """
    Author: Peppe Arrò
    This script calculates the 1D power spectrum for vector functions
    """
    if len(X.shape) == 1:
        Lx = X[-1] - X[0]
        Ly = Y[-1] - Y[0]
        x = X
        y = Y
    elif len(X.shape) == 2:
        Lx = X[-1,0] - X[0,0]
        Ly = Y[0,-1] - Y[0,0]
        x = X[:,0]
        y = Y[0,:]
    else:
        raise ValueError("X and Y must be 1D or 2D arrays")
    t = np.arange(field_x.shape[-1])
    nxc=len(X)-1
    nyc=len(Y)-1
    # Repeated boundaries must be excluded according to the definition of the FFT.
    field_x_ft=np.fft.rfft2(field_x[0:-1,0:-1,:],axes=(0,1))
    field_y_ft=np.fft.rfft2(field_y[0:-1,0:-1,:],axes=(0,1))
    field_z_ft=np.fft.rfft2(field_z[0:-1,0:-1,:],axes=(0,1))

    # 2D power spectrum.
    spec_2D=(abs(field_x_ft)**2+abs(field_y_ft)**2+abs(field_z_ft)**2)/((nxc*nyc)**2)
    spec_2D[:,1:-1,:]*=2 # Some modes are doubled to take into account the redundant ones removed by numpy's rfft.
    kx=np.fft.fftfreq(nxc-1,x[1])*2*np.pi
    ky=np.fft.rfftfreq(nyc-1,y[1])*2*np.pi

    # The 1D magnetic field energy spectrum is calculated.
    spec_1D=np.zeros((nxc//2+1,len(t)))

    for iy in range(len(ky)):
        for ix in range(len(kx)):
            try:
                index=round( np.sqrt( (Lx*kx[ix]/(2*np.pi))**2+(Ly*ky[iy]/(2*np.pi))**2 ) )
                if index<=(nxc//2):
                    spec_1D[index,:]+=spec_2D[ix,iy,:]
            except Exception as e:
                print(f"{index = }, {ix = }, {iy = }, {spec_2D.shape = }, {spec_1D.shape = }")
                raise e

    return ky,spec_1D #[:-1]


def get_spectral_index(k,spec,N):
	"""
        Calculate the spectral index by fitting a line to the log-log plot of the given spectrum.
        Parameters:
        k (array-like): The wavenumber array.
        spec (array-like): The spectrum array corresponding to the wavenumbers.
        N (int): The number of points to use in each segment for fitting.
        Returns:
        tuple: A tuple containing:
            - k_red (numpy.ndarray): The reduced wavenumber array, averaged over each segment.
            - slopes (numpy.ndarray): The slopes of the fitted lines, representing the spectral index for each segment.
        """
	from scipy.optimize import curve_fit
	
	def line(x,a,b):
		return a*x+b
	
	X=np.log10(k[1:])
	Y=np.log10(spec[1:])
	
	k_red=[]
	slopes=[]
	#print(k.shape,len(k)//N)
	for i in range(len(k)//N):
		#if i == 0:
		#	print(X[i*N:(i+1)*N].shape, Y[i*N:(i+1)*N].shape)
		p,_=curve_fit(line,X[i*N:(i+1)*N],Y[i*N:(i+1)*N],sigma=Y[i*N:(i+1)*N])
		k_red.append(np.mean(k[i*N+1:(i+1)*N+1]))
		slopes.append(p[0])
	
	return np.array(k_red), np.array(slopes)


def do_dot(fx,fy,fz,gx,gy,gz):
	return fx*gx+fy*gy+fz*gz
	
def do_cross(fx,fy,fz,gx,gy,gz):
	return fy*gz-fz*gy, fz*gx-fx*gz, fx*gy-fy*gx	

def get_PS_2D_field(data, x, y):
    """
    Get the pressure-strain term and theta
    """
    data['QJ'] = {}
    data['Qomega'] = {}
    data['QD'] = {}
    data['PiD'] = {}
    data['Ptheta'] = {}
    data['PS'] = {}
    data['theta'] = {}
    data['Dxx'] = {}
    data['Dyy'] = {}
    data['Dzz'] = {}
    data['Dxy'] = {}
    data['Dxz'] = {}
    data['Dyz'] = {}
    data['P'] = {}
    data['J*(E+VxB)'] = {}
    data['Jtotx'] = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
    data['Jtoty'] = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
    data['Jtotz'] = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
    E = np.array([data['Ex'], data['Ey'], data['Ez']]).transpose(1,2,3,0)
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
    J2 = data['Jtotx']**2 + data['Jtoty']**2 + data['Jtotz']**2
    data['QJ'] = 0.25*J2/np.mean(J2, axis=(0,1))
    for species in data['rho'].keys():
        J = np.array([data['Jx'][species], data['Jy'][species], data['Jz'][species]]).transpose(1,2,3,0)
        V = np.array([data['Vx'][species], data['Vy'][species], data['Vz'][species]]).transpose(1,2,3,0)
        data['J*(E+VxB)'][species] = np.sum(J*(E + np.cross(V, B)),axis=-1)
        uxx = np.gradient(data['Vx'][species],x, axis=0, edge_order=2)
        uxy = np.gradient(data['Vx'][species],y, axis=1, edge_order=2)
        uyx = np.gradient(data['Vy'][species],x, axis=0, edge_order=2)
        uyy = np.gradient(data['Vy'][species],y, axis=1, edge_order=2)
        uzx = np.gradient(data['Vz'][species],x, axis=0, edge_order=2)
        uzy = np.gradient(data['Vz'][species],y, axis=1, edge_order=2)
        omega2 = (uzy)**2 + (-uzx)**2 + (uyx-uxy)**2
        data['Qomega'][species] = 0.25*omega2/np.mean(omega2, axis=(0,1))
        data['P'][species]=(data['Pxx'][species]+\
                                data['Pyy'][species]+\
                                    data['Pzz'][species])/3
        data['theta'][species]=uxx+uyy
        data['PS'][species]=-data['Pxx'][species]*uxx-\
            data['Pxy'][species]*uxy-data['Pxy'][species]*uyx-\
                data['Pyy'][species]*uyy-data['Pxz'][species]*uzx-\
                    data['Pyz'][species]*uzy
        data['Ptheta'][species]=data['P'][species]*data['theta'][species]
        data['Dxx'][species] = uxx - data['theta'][species]/3
        data['Dyy'][species] = uyy - data['theta'][species]/3
        data['Dzz'][species] = -data['theta'][species]/3
        data['Dxy'][species] = (uxy + uyx)/2
        data['Dxz'][species] = uzx/2
        data['Dyz'][species] = uzy/2
        Dsum = data['Dxx'][species]**2 + data['Dyy'][species]**2 + data['Dzz'][species]**2 +\
            2*(data['Dxy'][species]**2 + data['Dxz'][species]**2 + data['Dyz'][species]**2) 
        data['QD'][species] = 0.25*Dsum/np.mean(Dsum, axis=(0,1))
        # Using PiD = - (Pij - Pdelta_ij)Dij
        data['PiD'][species]=-(data['Pxx'][species]-data['P'][species])*\
            (uxx-data['theta'][species]/3)-\
                (data['Pyy'][species]-data['P'][species])*\
                    (uyy-data['theta'][species]/3)-\
                        (data['Pzz'][species]-data['P'][species])*\
                            (-data['theta'][species]/3)-\
                                data['Pxy'][species]*(uyx+uxy)-\
                                    data['Pxz'][species]*(uzx)-\
                                        data['Pyz'][species]*(uzy)

def get_PS_2D(data, x, y):
    """
    Get the pressure-strain term and theta
    """
    for experiment in data.keys():
        get_PS_2D_field(data[experiment], x, y)

def apply_filter(field, density=None, filters = {'name': 'uniform_filter', 'size': 3, 'mode' : 'wrap', 'axes': (0,1)}):
    """
    Apply a specified filter to a given field, optionally using a density field for weighted filtering.
    Parameters:
    -----------
    field : numpy.ndarray
        The input field to which the filter will be applied.
    density : numpy.ndarray, optional
        The density field used for weighted filtering. If provided, it must have the same shape as `field` or be broadcastable to the shape of `field`.
    filters : dict, optional. A dictionary specifying the filter parameters. 
    Returns:
    --------
    numpy.ndarray
        The filtered field. If `density` is provided, the result is a density-weighted filtered field.
    Notes:
    ------
    - The filter function is dynamically retrieved from the `nd` module using the name provided in the `filters` dictionary.
    - If `density` is provided, the function performs a density-weighted filtering.
    - The filtering is applied only to the spatial dimensions specified by the `axes` parameter.
    """

    filters_copy = filters.copy()
    if not isinstance(filters_copy, dict):
        filters_object = getattr(nd, filters_copy)
    else:
        filters_name = filters_copy.pop("name", None)
        filters_object = getattr(nd, filters_name)
        filter_kwargs = filters_copy
        if isinstance(filter_kwargs['axes'], list):
            filter_kwargs['axes'] = tuple(filter_kwargs['axes'])
        if filter_kwargs['axes'] is None or filter_kwargs['axes'] != (0,1):
            print("Filtering targets should be aplied to only spatial dimensions")
    #print(filter_kwargs)
    if density is not None:
        if field.shape == density.shape:
            return filters_object(field*density, **filter_kwargs)/ filters_object(density, **filter_kwargs)
        else: #try to broadcast density assuming that field has one extra axes
            return filters_object(field*density[...,np.newaxis], **filter_kwargs)/ filters_object(density[...,np.newaxis], **filter_kwargs)
    else:
        return filters_object(field, **filter_kwargs)

def scale_filtering(data, x, y, qom, verbose=False,
                    filters = {'name': 'uniform_filter', 'size': 100, 'mode' : 'wrap', 'axes': (0,1)}):
    """
    Applies various filters to the input data and computes several derived quantities.
    Parameters:
    data (dict): A dictionary containing the experimental data. 
    Returns:
    None: The function modifies the input data dictionary in place by adding filtered and derived quantities.
    Notes:
    - The function computes filtered versions of the magnetic and electric fields.
    - It computes Favre-averaged quantities for velocity components.
    - It calculates the energy densities (kinetic and magnetic).
    - It computes various pressure and interaction terms based on the filtered data.
    - The function assumes the existence of certain keys and structures within the input data dictionary.
    """
    auxiliary = {} 
    for fields in ['Bx', 'By', 'Bz', 'Ex','Ey', 'Ez']:
        auxiliary[f"{fields}_bar"] = apply_filter(data[fields], filters = filters)
      
    for fields in ['Vx', 'Vy', 'Vz', 'Bx', 'By', 'Bz', 'Ex','Ey', 'Ez']:
        auxiliary[f"{fields}_favre"] = {}
        
    data['E2_bar'] = (auxiliary['Ex_bar']**2 + auxiliary['Ey_bar']**2 + auxiliary['Ez_bar']**2)/(8*np.pi)
    data['B2_bar'] = (auxiliary['Bx_bar']**2 + auxiliary['By_bar']**2 + auxiliary['Bz_bar']**2)/(8*np.pi)
    data['Ef_favre'] = {}
    data['PIuu'] = {}
    data['PIbb'] = {}
    data['PS'] = {}
    data['-Ptheta'] = {}
    data['JdotE'] = {}
    auxiliary['rho_bar'] = {}
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
    E_bar = np.array([auxiliary['Ex_bar'], auxiliary['Ey_bar'], auxiliary['Ez_bar']]).transpose(1,2,3,0)
    for i, species in enumerate(data['rho'].keys()):
        for fields in ['Vx', 'Vy', 'Vz']:
            auxiliary[f"{fields}_favre"][species] = apply_filter(data[fields][species], density=data['rho'][species], filters = filters)
        for fields in ['Bx', 'By', 'Bz', 'Ex','Ey', 'Ez']:
            auxiliary[f"{fields}_favre"][species] = apply_filter(data[fields], density=data['rho'][species], filters = filters)
        auxiliary['rho_bar'][species] = apply_filter(data['rho'][species], filters = filters) # charge density
        data['Ef_favre'][species] = 0.5*auxiliary['rho_bar'][species]*(auxiliary['Vx_favre'][species]**2 + auxiliary['Vy_favre'][species]**2 + auxiliary['Vz_favre'][species]**2)/qom[i]
        B_favre = np.array([auxiliary['Bx_favre'][species], auxiliary['By_favre'][species], auxiliary['Bz_favre'][species]]).transpose(1,2,3,0)
        E_favre = np.array([auxiliary['Ex_favre'][species], auxiliary['Ey_favre'][species], auxiliary['Ez_favre'][species]]).transpose(1,2,3,0)
        tau_e = E_favre - E_bar
        if verbose:
            print(f"{species = }")
        V_favre = np.array([auxiliary['Vx_favre'][species], auxiliary['Vy_favre'][species], auxiliary['Vz_favre'][species]]).transpose(1,2,3,0)
        data['PIbb'][species] = -auxiliary['rho_bar'][species]*np.sum(tau_e*V_favre, axis=-1)
        data['JdotE'][species] = +auxiliary['rho_bar'][species]*np.sum(E_favre*V_favre, axis=-1)

        V = np.array([data['Vx'][species], data['Vy'][species], data['Vz'][species]]).transpose(1,2,3,0)
        tau_b = apply_filter(np.cross(V, B), density=data['rho'][species], filters = filters) - np.cross(V_favre, B_favre)
        dV_favre = {}
        for component in ['x', 'y', 'z']:
            dV_favre[f"{component}x"] = np.gradient(auxiliary[f'V{component}_favre'][species],x, axis=0, edge_order=2)
            dV_favre[f"{component}y"] = np.gradient(auxiliary[f'V{component}_favre'][species],y, axis=1, edge_order=2)
        data['-Ptheta'][species] = 0
        for component in ['x', 'y', 'z']: #calculating trace
            data['-Ptheta'][species] += apply_filter(data[f'P{component}{component}'][species], filters = filters)
        data['-Ptheta'][species] *= -(dV_favre['xx']+dV_favre['yy'])/3 # divergence of velocity times pressure trace
        data['PIuu'][species] = 0 # equation (21) of Matthaeus, W. H.; Yang, Y.; Wan, M.; Parashar, T. N.; Bandyopadhyay, R.; Chasapis, A.; Pezzi, O.; Valentini, F. Pathways to Dissipation in Weakly Collisional Plasmas. ApJ 2020, 891 (1), 101. https://doi.org/10.3847/1538-4357/ab6d6a. See also Yang, Y.; Matthaeus, W. H.; Roy, S.; Roytershteyn, V.; Parashar, T. N.; Bandyopadhyay, R.; Wan, M. Pressure–Strain Interaction as the Energy Dissipation Estimate in Collisionless Plasma. ApJ 2022, 929 (2), 142. https://doi.org/10.3847/1538-4357/ac5d3e.
        data['PS'][species] = 0
        for component1, component2 in zip(['x', 'x', 'y' ,'y', 'z', 'z'], ['x', 'y', 'x', 'y', 'x', 'y']):
            Pbar = apply_filter(data[f'P{component1}{component2}'][species], filters = filters)
            if verbose:
                print(f"adding: Pbar{component1}{component2} * nabla dVfavre_{component1}d{component2}")
            data['PS'][species] += - Pbar*dV_favre[f"{component1}{component2}"]

            tauu = apply_filter(data[f'V{component1}'][species]*data[f'V{component2}'][species], \
                            density=data['rho'][species], filters = filters) - \
                            auxiliary[f'V{component1}_favre'][species]*auxiliary[f'V{component2}_favre'][species]

            data['PIuu'][species] += - auxiliary['rho_bar'][species]*tauu*dV_favre[f"{component1}{component2}"]/qom[i]
            
        data['PIuu'][species] += - auxiliary['rho_bar'][species]*np.sum(tau_b*V_favre, axis=-1)


def get_T(data, qom):
    """
    Get T, T_perp, T_par
    """
    data['T'] = {}
    data['T_par'] = {}
    data['T_perp'] = {}
    data['beta_par'] = {}
    bx=data['Bx']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
    by=data['By']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
    bz=data['Bz']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
    for i, species in enumerate(data['rho'].keys()):
        data['T'][species]=(data['Pxx'][species]+\
                                data['Pyy'][species]+\
                                    data['Pzz'][species])/(3*data['rho'][species]*np.sign(qom[i]))
        data['T_par'][species]=(data['Pxx'][species]*bx**2+\
            data['Pyy'][species]*by**2+data['Pzz'][species]*bz**2+\
                2*(data['Pxy'][species]*bx*by+data['Pxz'][species]*bx*bz+\
                    data['Pyz'][species]*by*bz))/(data['rho'][species]*np.sign(qom[i]))
        data['T_perp'][species]=(3*data['T'][species]-data['T_par'][species])/2
        data['beta_par'][species] = 8*np.pi*data['T_par'][species]*(data['rho'][species]*np.sign(qom[i]))/(data['Bx']**2 + data['By']**2 + data['Bz']**2)

def get_agyrotropy(data):
	for experiment in data.keys():
		data['agyrotropy'] = {}
		for species in data['rho'].keys():
			bx=data['Bx']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
			by=data['By']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
			bz=data['Bz']/np.sqrt(data['Bx']**2+data['By']**2+data['Bz']**2)
			I1=data['Pxx'][species]+data['Pyy'][species]+data['Pzz'][species]
			I2=data['Pxx'][species]*data['Pyy'][species]+\
				data['Pxx'][species]*data['Pzz'][species]+\
					data['Pyy'][species]*data['Pzz'][species]-\
						(data['Pxy'][species]**2+data['Pxz'][species]**2+\
	   data['Pyz'][species]**2)
			P_par=data['Pxx'][species]*bx**2+data['Pyy'][species]*by**2+\
				data['Pzz'][species]*bz**2+2*(data['Pxy'][species]*bx*by+\
											  data['Pxz'][species]*bx*bz+\
												data['Pyz'][species]*by*bz)
			data['agyrotropy'][species]=1-4*I2/((I1-P_par)*(I1+3*P_par))

def highdiff(data, dx, dy, coeff = None, axis=0, **kwargs):
    """
    Compute the 4th-order central finite difference derivative for a 2D array 
    along either the x or y axis.

    Parameters:
        data (ndarray): Input 2D or higher-dimensional array.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        coeff (ndarray): Coefficients for the finite difference scheme.
            Default is 4th-order central difference coefficients.
        axis (str): Axis along which to compute the derivative (0 or 1).

    Returns:
        ndarray: The derivative along the specified axis.
    """
    # 4th-order finite difference coefficients
    if coeff is None:
        coeff = np.array([-1, 8, 0, -8, 1]) / 12.0
    
    if axis == 0:
        # Compute derivative along the x-axis
        dx_kernel = coeff.reshape((-1,) + (1,) * (data.ndim - 1))  # generalizing reshape (-1,1) for higher dimensions
        return nd.convolve(data, dx_kernel, output=float, **kwargs) / dx
    elif axis == 1:
        # Compute derivative along the y-axis
        dy_kernel = coeff.reshape((1, -1) + (1,) * (data.ndim - 2))   # generalizing reshape (1,-1) for higher dimensions
        return nd.convolve(data, dy_kernel, output=float, **kwargs) / dy
    else:
        raise ValueError("Invalid axis. Use 0 or 1.")

def get_Ohm(data,qom, x,y, coeff=None):
    """
    Compute the electric field and other derived quantities based on the input data.
    This function calculates the electric field, ExB/B^2, EHall, EMHD, and other quantities
    using the provided data dictionary. It also computes the pressure gradient and other
    relevant quantities based on the input data.
    """
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
    E = np.array([data['Ex'], data['Ey'], data['Ez']]).transpose(1,2,3,0)
    data['ExB/B^2'] = np.cross(E,B)/(data['Bx']**2+data['By']**2+data['Bz']**2)[...,np.newaxis]
    data['Jtotx'] = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
    data['Jtoty'] = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
    data['Jtotz'] = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
    J = np.array([data['Jtotx'], data['Jtoty'], data['Jtotz']]).transpose(1,2,3,0)
    data['EHall_x'], data['EHall_y'], data['EHall_z'] = (np.cross(J,B)/(-data['rho']['e'])[...,np.newaxis]).transpose(3,0,1,2)
    norm = 0
    data['uCMx'] = 0
    data['uCMy'] = 0
    data['uCMz'] = 0
    for i, species in enumerate(data['rho'].keys()):
        data['uCMx'] += (data['rho'][species]/qom[i])*data['Vx'][species]
        data['uCMy'] += (data['rho'][species]/qom[i])*data['Vy'][species]
        data['uCMz'] += (data['rho'][species]/qom[i])*data['Vz'][species]
        norm += data['rho'][species]/qom[i]
    data['uCMx'] /= norm
    data['uCMy'] /= norm
    data['uCMz'] /= norm
    uCM = np.array([data['uCMx'], data['uCMy'], data['uCMz']]).transpose(1,2,3,0)
    data['EMHD_x'], data['EMHD_y'], data['EMHD_z'] = - np.cross(uCM,B).transpose(3,0,1,2)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    #data['EP_x'] = (np.gradient(data['Pxx']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pxy']['e'],y,axis=1,edge_order=2))/data['rho']['e']
    #data['EP_y'] = (np.gradient(data['Pxy']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pyy']['e'],y,axis=1,edge_order=2))/data['rho']['e']
    #data['EP_z'] = (np.gradient(data['Pxz']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pyz']['e'],y,axis=1,edge_order=2))/data['rho']['e']
    data['EP_x'] = -(highdiff(data['Pxx']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap') + highdiff(data['Pxy']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
    data['EP_y'] = -(highdiff(data['Pxy']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap') + highdiff(data['Pyy']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
    data['EP_z'] = -(highdiff(data['Pxz']['e'], dx, dy, coeff=coeff, axis=0, mode='wrap') + highdiff(data['Pyz']['e'], dx, dy, coeff=coeff, axis=1, mode='wrap'))/(-data['rho']['e']) # density in ECsim is negative (electron charge density)
    


def get_Az(x,y,data):
    def get_Az(x, y, data):
        """
        Compute the vector potential component Az based on the input magnetic field components Bx and By.
        This function calculates the Az component of the vector potential using the provided
        magnetic field data (Bx and By) and spatial coordinates (x and y). The calculation
        is performed using numerical integration along the x and y axes.
        Parameters:
        -----------
        x : numpy.ndarray
            1D array representing the x-coordinates of the grid points.
        y : numpy.ndarray
            1D array representing the y-coordinates of the grid points.
        data : dict
            Dictionary containing the magnetic field components:
            - 'Bx': 3D numpy array representing the x-component of the magnetic field.
            - 'By': 3D numpy array representing the y-component of the magnetic field.
        Modifies:
        ---------
        data : dict
            Adds a new key 'Az' to the input dictionary, which contains the computed
            3D numpy array of the Az component of the vector potential.
        Notes:
        ------
        - The function assumes that the input magnetic field components ('Bx' and 'By')
          are defined on a regular grid.
        - The grid spacing is computed as the difference between consecutive elements
          in the x and y arrays (dx and dy).
        - The integration is performed using a trapezoidal rule along the respective axes.
        Example:
        --------
        >>> ut.get_Az(X[:,0],Y[0,:],data)
        >>> print(data['Az'])  # Access the computed Az component
        """
    
    Nx=data['Bx'].shape[0]
    Ny=data['Bx'].shape[1]
    Nz=data['Bx'].shape[2]
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    f=np.zeros((Nx,Ny,Nz))
    g=np.zeros((Nx,Ny,Nz))
    
    for iy in range(1,Ny):
        g[:,iy,:]=g[:,iy-1,:]+(data['Bx'][:,iy-1,:]+data['Bx'][:,iy,:])*dy/2
        
    for iy in range(0,Ny):
        for ix in range(1,Nx):
            f[ix,iy,:]=f[ix-1,iy,:]-(data['By'][ix-1,0,:]+data['By'][ix,0,:])*dx/2    
    data['Az'] = f+g

    



def get_W(data):
	"""
	Get W
	"""
	for experiment in data.keys():
		data[experiment]['W'] = {}
		for species in data[experiment]['rho'].keys():
			data[experiment]['W'][species] = do_dot(data[experiment]['Ex'],data[experiment]['Ey'],data[experiment]['Ez'],\
						data[experiment]['Jx'][species],data[experiment]['Jy'][species],data[experiment]['Jz'][species])

def get_D(data):
	"""
	Get D
	"""
	for experiment in data.keys():
		data[experiment]['D'] = {}
		for species in data[experiment]['rho'].keys():
			data[experiment]['D'][species] = do_dot(data[experiment]['Jx'][species],data[experiment]['Jy'][species],data[experiment]['Jz'][species],\
						data[experiment]['Jx'][species],data[experiment]['Jy'][species],data[experiment]['Jz'][species])


