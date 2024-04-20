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