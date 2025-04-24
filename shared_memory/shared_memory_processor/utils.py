# shared_memory_processor/utils.py
import os
import numpy as np

def get_script_dir():
    """
    Returns the directory of the current script.
    """
    return os.path.dirname(os.path.abspath(__file__))

def should_serialize(value):
    """
    Determines if the value should be serialized (e.g. via JSON)
    because it cannot be converted to a uniform numpy array.
    """
    # Already a numpy array? Use it directly.
    if isinstance(value, np.ndarray):
        return False
    # If it's a list, try converting and inspect the resulting dtype.
    if isinstance(value, list):
        try:
            arr = np.array(value)
            # If conversion yields an object dtype, the list is non-uniform.
            return arr.dtype == 'object'
        except Exception:
            return True
    # Dictionaries are always treated as complex.
    if isinstance(value, dict):
        return True
    # Otherwise, assume it's simple.
    return False