# shared_memory_processor/function_registry.py
import inspect, types, importlib.util, os
from shared_memory_processor.decorators import classproperty

class Functions:
    """
    Registry class for managing functions.
    
    Functions can be dynamically added and later retrieved as a dictionary.
    """
    
    IGNORE_LIST = ['get_function_list', 'add_function', 'njit','least_squares','wraps']
    
    @classmethod
    def get_function_list(cls):
        """
       Retrieve all registered functions excluding those in the ignore list.
       
       Returns:
           dict: Mapping of function names to function objects.
       """
        functions = {}
        # Use dir() to get all attributes; descriptors are automatically called
        for name in dir(cls):
            if name in cls.IGNORE_LIST:
                continue  # Skip ignored names
            attr = getattr(cls, name)
            
            # Check if attribute is a function or method
            if isinstance(attr, (types.FunctionType, types.MethodType)):
                functions[name] = attr
        return functions
    
    @classmethod
    def add_function(cls, name, func, method_type='instance'):
        """
        Add a new function to the Functions registry.
        
        Args:
            name (str): Name of the function.
            func (callable): The function object to add.
            method_type (str): Type of method ('instance', 'classmethod', or 'staticmethod').
        """
        if method_type == 'classmethod':
            func = classmethod(func)
        elif method_type == 'staticmethod':
            func = staticmethod(func)
            
        # Dynamically attach the function to the class
        setattr(cls, name, func)

def import_functions(custom_dir, target_class, method_type='staticmethod'):
    """
    Import functions from Python files in a given directory and register them.
    
    Scans each .py file (excluding files starting with '_') in custom_dir,
    loads the module, and registers all found functions with target_class.
    
    Args:
        custom_dir (str): Directory path to search for Python files.
        target_class (type): Class to which the functions will be added.
        method_type (str): Method type for the added functions 
                           ('instance', 'classmethod', or 'staticmethod').
    """
    for filename in os.listdir(custom_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]  # Remove '.py' extension
            module_path = os.path.join(custom_dir, filename)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Register each function found in the module
            for name, func in inspect.getmembers(module, predicate=inspect.isfunction):
                target_class.add_function(name, func, method_type=method_type)