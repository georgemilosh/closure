import subprocess


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


# TODO: write a function that plots the predicted vs actual values of a regression model