edit: line 281 shared_memory.py

#return np.ascontiguousarray(np.reshape(np.transpose(data, new_dimensions1), new_dimensions2, 'C'))
return np.ascontiguousarray(np.reshape(np.transpose(data, new_dimensions1), new_dimensions2, order='C'))

edit: line 24 shared_memory.py
#LIBRARY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "shared_memory.dll")
LIBRARY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "shared_memory.dll")

# Shared Memory Processor

Shared Memory Processor is a package based on ben-izd's shared_memory module (https://github.com/ben-izd/shared_memory/). A hybrid MATLAB-Python package that provides a framework for processing and synchronizing shared memory segments, designed to facilitate efficient data handling, dynamic function registration, and synchronization of data processing across systems. 

## Features

- **Shared Memory Management:**  
  Create, read, and update shared memory segments using a dedicated processor.
  
- **Function Synchronization:**  
  Use decorators to ensure shared memory is synchronized before and after function execution.
  
- **Dynamic Function Registration:**  
  Automatically import and register custom functions from a specified directory.
  
- **MATLAB Integration:**  
  Designed to work with MATLAB, ensuring proper data formats and efficient memory usage.

## Package Structure

- **`processor.py`**  
  Contains the `SHMEM_Processor` class, which handles clearing buffers, reading metadata, and creating/updating shared memory segments.

- **`decorators.py`**  
  Provides decorators such as `sync_shared_memory` (which synchronizes shared memory with function calls) and a `classproperty` descriptor.

- **`function_registry.py`**  
  Manages dynamic function registration. It scans a custom directory for Python files, imports functions, and registers them with a central registry.

- **`utils.py`**  
  Contains utility functions (e.g., `get_script_dir`) to assist with file path operations and other common tasks.

- **`__init__.py`**  
  Initializes the package by exposing key classes and functions. It also imports custom functions from a `custom_functions` directory and registers them automatically.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://your.repo.url/shared_memory_processor.git
    cd shared_memory_processor
    ```

2. **Install Dependencies:**

    Ensure you have Python installed along with necessary packages such as NumPy. For example:

    ```bash
    pip install numpy
    ```

3. **Set Up Shared Memory Directory:**

    The package expects a `shared_memory-main` directory in a specific relative location. Ensure that this directory is correctly placed.

