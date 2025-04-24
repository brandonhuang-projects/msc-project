# shared_memory_processor/decorators.py
from functools import wraps
import numpy as np
from .utils import should_serialize

class classproperty:
    """
    Descriptor for defining properties on the class itself.
    
    Allows methods to be accessed as class-level properties.
    """
    def __init__(self, fget):
        self.fget = fget   # Store the getter function
    def __get__(self, instance, owner):
        # Call the getter with the class as the argument
        return self.fget(owner)

def sync_shared_memory(processor, func):
    """
    Decorator to synchronize shared memory with function calls.
    
    Reads shared memory data via the processor, processes it,
    passes it as arguments to the wrapped function, then updates
    the shared memory with the function's result.
    
    Args:
        processor: Object providing shared memory methods.
        func (callable): Function that uses shared memory data as its arguments.
    
    Returns:
        The result from the wrapped function after updating shared memory.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        # Retrieve shared memory data using the processor
        shared_memory_data = processor._read_shared_memory_metadata()
        
        shared_memory_data = clean_input(shared_memory_data)
        
        # Call the original function with shared memory data
        result = func(*shared_memory_data)
        
        result = clean_output(result)
        
        # Update the shared memory segments with the processed result
        processor._create_shared_memory_segments(*result)
        
        return result
    return wrapper

def clean_input(shared_memory_data):
    
    for index, item in enumerate(shared_memory_data):      
        
        # Remove singleton dimensions.
        if hasattr(item, 'squeeze'):
            try:
                item = item.squeeze()
            except Exception:
                pass
    
        # Convert it to a native Python scalar
        if hasattr(item, 'item'):
            try:
                item = item.item()
            except Exception:
                pass
            
        shared_memory_data[index] = item
        
    return shared_memory_data
    
def clean_output(result):        
    # Ensure function result is at least a list
    if isinstance(result, np.ndarray) or not isinstance(result, (list, tuple)):
        result = [result]
    else:
        if type(result) != str:
            result = list(result)
    
    # Convert each item to a 1D numpy array
    for index, item in enumerate(result):
            
        if type(item) == str or type(item) == dict:
            continue  # Skip strings
        try:
            result[index] = np.atleast_1d(np.asarray(item))
        except ValueError:
            if should_serialize(item):
                result[index] = [i.tolist() for i in item]
                
    return result