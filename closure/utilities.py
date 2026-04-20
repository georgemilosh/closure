"""Backward-compatible helpers and lightweight utility functions.

The historical ``utilities`` module mixed together configuration, ML,
plotting, and plasma-physics code. Those larger responsibilities now live in
``closure.config``, ``closure.evaluation``, ``closure.visualization``, and
``closure.plasma``. The functions kept here are either small generic helpers or
lazy compatibility wrappers for the old import paths.
"""

from __future__ import annotations

__all__ = [
    "alias",
    "append_index_to_duplicates",
    "compare_metrics",
    "compare_runs",
    "compute_loss",
    "conserved_quantities",
    "evaluate_loss",
    "get_duplicate_indices",
    "get_git_revision_hash",
    "graph_pred_targets",
    "load_and_compute_difference",
    "normalize_input",
    "parse_score",
    "plot_pred_targets",
    "pred_ground_targets",
    "pred_unnormalize",
    "prediction2data",
    "set_nested_config",
    "species_to_list",
    "transform_features",
    "transform_targets",
    "unnormalize_output",
    "apply_filter",
    "code2alfven",
    "do_cross",
    "do_dot",
    "get_Az",
    "get_D",
    "get_J_perp",
    "get_Ohm",
    "get_PS_2D",
    "get_PS_2D_field",
    "get_PS_3D_field",
    "get_T",
    "get_W",
    "get_agyrotropy",
    "get_spectral_index",
    "highdiff",
    "scalar_spectrum_2D",
    "scale_filtering",
    "vector_spectrum_2D",
]

import pickle
import subprocess
from typing import Any, Callable

import pandas as pd

def alias(*names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    A decorator that assigns multiple global aliases to a function. It allows conveniently renaming functions without breaking backward compatibility
    Args:
        *names: One or more strings representing the alias names to assign to the decorated function.
    Returns:
        decorator: A decorator that, when applied to a function, adds the function to the global namespace under each specified alias.
    Example:
        @alias('foo', 'bar')
        def my_function():
            pass
        # Now, my_function can be accessed as foo or bar in the global namespace.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        globals_ = globals()
        for name in names:
            globals_[name] = func
        return func
    return decorator


def _lazy_import(module_name: str, attr_name: str) -> Any:
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)

def set_nested_config(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible wrapper around :func:`closure.config.set_nested_config`."""
    return _lazy_import("closure.config", "set_nested_config")(*args, **kwargs)


def species_to_list(input_list: list[str]) -> list[str | list[str]]:
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


def load_and_compute_difference(file_path: str) -> dict[str, Any]:
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

def append_index_to_duplicates(lst: list[Any]) -> list[Any]:
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

def get_duplicate_indices(lst: list[Any]) -> dict[Any, list[int]]:
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

def parse_score(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "parse_score")(*args, **kwargs)

def compare_runs(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "compare_runs")(*args, **kwargs)

def compare_metrics(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "compare_metrics")(*args, **kwargs)

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
    data = pd.read_csv(file_path,  delim_whitespace=True, 
                    comment='#', 
                    header=None)
    # Select only the first len(column_names) columns and assign the names
    data = data.iloc[:, :len(column_names)]
    data.columns = column_names
    if verbose:
        print("variables ", column_names[1:])
    
    return data, column_names

def transform_features(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "transform_features")(*args, **kwargs)

def transform_targets(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "transform_targets")(*args, **kwargs)

def compute_loss(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "compute_loss")(*args, **kwargs)

def evaluate_loss(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "evaluate_loss")(*args, **kwargs)

def graph_pred_targets(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.visualization", "graph_pred_targets")(*args, **kwargs)

def pred_ground_targets(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "pred_ground_targets")(*args, **kwargs)

def plot_pred_targets(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.visualization", "plot_pred_targets")(*args, **kwargs)

def normalize_input(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "normalize_input")(*args, **kwargs)

def pred_unnormalize(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "pred_unnormalize")(*args, **kwargs)

unnormalize_output = pred_unnormalize  # backward compat alias

def prediction2data(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.evaluation", "prediction2data")(*args, **kwargs)

def highdiff(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "highdiff")(*args, **kwargs)


def do_dot(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "do_dot")(*args, **kwargs)


def do_cross(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "do_cross")(*args, **kwargs)


def get_Ohm(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_Ohm")(*args, **kwargs)


def get_PS_2D_field(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_PS_2D_field")(*args, **kwargs)


def get_PS_3D_field(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_PS_3D_field")(*args, **kwargs)


def get_PS_2D(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_PS_2D")(*args, **kwargs)


def get_Az(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_Az")(*args, **kwargs)


def get_J_perp(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_J_perp")(*args, **kwargs)


def get_W(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_W")(*args, **kwargs)


def get_D(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_D")(*args, **kwargs)


def get_agyrotropy(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_agyrotropy")(*args, **kwargs)


def get_T(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_T")(*args, **kwargs)


def code2alfven(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "code2alfven")(*args, **kwargs)


def apply_filter(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "apply_filter")(*args, **kwargs)


def scale_filtering(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "scale_filtering")(*args, **kwargs)


def scalar_spectrum_2D(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "scalar_spectrum_2D")(*args, **kwargs)


def vector_spectrum_2D(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "vector_spectrum_2D")(*args, **kwargs)


def get_spectral_index(*args: Any, **kwargs: Any) -> Any:
    return _lazy_import("closure.plasma", "get_spectral_index")(*args, **kwargs)


