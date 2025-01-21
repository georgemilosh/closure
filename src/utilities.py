import subprocess
from . import trainers as tr
import pandas as pd
import torch
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
from . import read_pic as rp
import re
import os
import scipy.ndimage as nd

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

    for work_dir, run in zip(work_dirs,runs):
        if not os.path.exists(work_dir):
            raise ValueError(f"Work directory '{work_dir}' does not exist.")
        trainer = tr.Trainer(work_dir=work_dir, **kwargs)
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
    
    ground_truth = trainer.test_dataset.features[:,trainer.val_loader.feature_channels].squeeze()
    pred_shape = [1 for _ in ground_truth.cpu().numpy().shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if trainer.train_loader.feature_channels is None: 
        list_of_feature_indices = range(len(trainer.dataset_kwargs['read_features_targets_kwargs']['request_features']))
    else:
        list_of_feature_indices = trainer.train_loader.feature_channels

    if renorm:
        ground_truth_scaled = (ground_truth*trainer.test_dataset.features_std[list_of_feature_indices].reshape(pred_shape)+
                                trainer.test_dataset.features_mean[list_of_feature_indices].reshape(pred_shape))
    if rescale:
        for channel, _ in enumerate(trainer.train_dataset.request_features):
            if trainer.train_loader.feature_channels is None:
                list_of_feature_indices = range(len(trainer.dataset_kwargs['read_features_targets_kwargs']['request_features']))
            else:
                list_of_feature_indices = trainer.train_loader.feature_channels
            if trainer.train_dataset.prescaler_features is not None:
                func = [trainer.train_dataset.prescaler_features[i] for i in list_of_feature_indices][channel]
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
    ground_truth = trainer.test_dataset.targets[:,trainer.val_loader.target_channels].squeeze()
    pred_shape = [1 for _ in prediction.shape]
    pred_shape[1] = -1
    pred_shape = tuple(pred_shape)

    if trainer.train_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.train_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.train_loader.target_channels

    if renorm:
        prediction_scaled = (prediction*trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)+
                            trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape))
        ground_truth_scaled = (ground_truth*trainer.test_dataset.targets_std[list_of_target_indices].reshape(pred_shape)+
                                trainer.test_dataset.targets_mean[list_of_target_indices].reshape(pred_shape))
    if rescale:
        for channel, _ in enumerate(trainer.train_dataset.request_targets):
            if trainer.train_loader.target_channels is None:
                list_of_target_indices = range(len(trainer.train_dataset.prescaler_targets))
            else:
                list_of_target_indices = trainer.train_loader.target_channels

            func = [trainer.train_dataset.prescaler_targets[i] for i in list_of_target_indices][channel]
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
        loss = (1- torch.nn.MSELoss()(ground_truth,prediction)/torch.var(ground_truth)).cpu().numpy()
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
    if trainer.train_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.train_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.train_loader.target_channels
    for channel in list_of_target_indices:
        label = f'{trainer.train_dataset.request_targets[channel]}_{criterion}'
        loss[label] = compute_loss(ground_truth[:,channel].flatten(),prediction[:,channel].flatten(),criterion)
        if verbose:
            print(f'Loss for channel {channel}:  {trainer.train_dataset.request_targets[channel]}, loss = {loss[label]}')
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
    channel = trainer.train_dataset.request_targets.index(target_name)
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
    ground_truth = trainer.test_dataset.targets[:,trainer.val_loader.target_channels].squeeze()
    loss = trainer.model._compute_loss(ground_truth.flatten(),prediction.flatten(),trainer.model.criterion)
    if verbose:
        print(f"Total loss {loss}")
    if trainer.train_loader.target_channels is None:
        list_of_target_indices = range(len(trainer.train_dataset.prescaler_targets))
    else:
        list_of_target_indices = trainer.train_loader.target_channels
    for channel in list_of_target_indices:
        try:
            loss = trainer.model._compute_loss(ground_truth[:,channel].flatten(),prediction[:,channel].flatten(),trainer.model.criterion)
        except Exception as e:
            print(f"{ground_truth.shape = }, {prediction.shape = }, {channel = }, {trainer.model.criterion = }")
            raise e
        if verbose:
            print(f'Loss for channel {channel}:  {trainer.train_dataset.request_targets[channel]}, loss = {loss}')
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
    print("The function pred_ground_targets is deprecated. Use graph_pred_targets instead.")
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



def get_Ohm(data,qom, x,y):
    """
    ExB/B^2

    Notice that if electrons are massless qom = np.inf
    """
    B = np.array([data['Bx'], data['By'], data['Bz']]).transpose(1,2,3,0)
    E = np.array([data['Ex'], data['Ey'], data['Ez']]).transpose(1,2,3,0)
    data['ExB/B^2'] = np.cross(E,B)/(data['Bx']**2+data['By']**2+data['Bz']**2)[...,np.newaxis]
    data['Jtotx'] = np.sum([data['Jx'][species] for species in data['Jx'].keys()], axis=0)
    data['Jtoty'] = np.sum([data['Jy'][species] for species in data['Jy'].keys()], axis=0)
    data['Jtotz'] = np.sum([data['Jz'][species] for species in data['Jz'].keys()], axis=0)
    J = np.array([data['Jtotx'], data['Jtoty'], data['Jtotz']]).transpose(1,2,3,0)
    data['EHall_x'], data['EHall_y'], data['EHall_z'] = - (np.cross(J,B)/(data['rho']['e'])[...,np.newaxis]).transpose(3,0,1,2)
    norm = 0
    for i, species in enumerate(data['rho'].keys()):
        if 'uCMx' in data.keys():
            data['uCMx'] += (data['rho'][species]/qom[i])*data['Vx'][species]
            data['uCMy'] += (data['rho'][species]/qom[i])*data['Vy'][species]
            data['uCMz'] += (data['rho'][species]/qom[i])*data['Vz'][species]
        else:
            data['uCMx'] = (data['rho'][species]/qom[i])*data['Vx'][species]
            data['uCMy'] = (data['rho'][species]/qom[i])*data['Vy'][species]
            data['uCMz'] = (data['rho'][species]/qom[i])*data['Vz'][species]
        norm += data['rho'][species]/qom[i]
    data['uCMx'] /= norm
    uCM = np.array([data['uCMx'], data['uCMy'], data['uCMz']]).transpose(1,2,3,0)
    data['EMHD_x'], data['EMHD_y'], data['EMHD_z'] = - np.cross(uCM,B).transpose(3,0,1,2)

    data['EP_x'] = (np.gradient(data['Pxx']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pxy']['e'],y,axis=1,edge_order=2))/data['rho']['e']
    data['EP_y'] = (np.gradient(data['Pxy']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pyy']['e'],y,axis=1,edge_order=2))/data['rho']['e']
    data['EP_z'] = (np.gradient(data['Pxz']['e'],x,axis=0,edge_order=2)+np.gradient(data['Pyz']['e'],y,axis=1,edge_order=2))/data['rho']['e']





    



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


