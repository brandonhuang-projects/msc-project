# shared_memory_processor/__init__.py
from .processor import SHMEM_Processor
from .decorators import classproperty, sync_shared_memory
from .function_registry import Functions, import_functions
from .utils import get_script_dir

__all__ = [
    "SHMEM_Processor",
    "classproperty",
    "sync_shared_memory",
    "Functions",
    "import_functions",
    "get_script_dir"
]

import os
current_dir = os.path.dirname(get_script_dir())
custom_functions_dir = os.path.join(current_dir, "custom_functions")

# Import and register functions
import_functions(custom_functions_dir, Functions, method_type='staticmethod')