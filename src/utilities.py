import subprocess
from . import trainers as tr
import pandas as pd
import torch
import torchmetrics

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
    loss_df = None

    for work_dir, run in zip(work_dirs,runs):
        trainer = tr.Trainer(work_dir=work_dir)
        trainer.load_run(run)
        prediction = trainer.model.predict(trainer.test_dataset.features)
        ground_truth = trainer.test_dataset.targets[:,trainer.val_loader.target_channels].squeeze()
        # computing total loss
        total_loss = trainer.model._compute_loss(ground_truth,prediction,trainer.model.criterion).cpu().numpy()
        score = {}
        if metric is not None:
            for metric_name in metric:
                score[f"total_{metric_name}"] = trainer.model._compute_loss(ground_truth,prediction,
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
            target_loss = trainer.model._compute_loss(ground_truth[:, channel], prediction[:, channel], trainer.model.criterion)
            loss_dict[trainer.train_dataset.request_targets[channel]] = target_loss.cpu().numpy()
            if metric is not None:
                for metric_name in metric:
                    loss_dict[f"{trainer.train_dataset.request_targets[channel]}_{metric_name}"] = \
                        trainer.model._compute_loss(ground_truth[:, channel], prediction[:, channel],
                                                                 parse_score(metric_name)).cpu().numpy()

        if loss_df is None:
            loss_df = pd.DataFrame(columns=loss_dict.keys())
        loss_df.loc[len(loss_df)] = loss_dict

    return loss_df