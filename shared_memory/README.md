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

## Usage

### Initializing the Processor

```python
from shared_memory_processor import SHMEM_Processor

processor = SHMEM_Processor()
Synchronizing Shared Memory with Functions
Decorate a function so that it automatically synchronizes shared memory data:

python
Copy
from shared_memory_processor import sync_shared_memory

@sync_shared_memory(processor, your_function)
def your_function(data1, data2):
    # Process the shared memory data
    result = ...  # your processing logic here
    return [result]  # Ensure the output is wrapped in a list
Running a Registered Function
An example of how you might run a function (e.g., "whittaker") with the processor:

matlab
Copy
% MATLAB example calling the Python-based processor:
app.Base = runProcessor('whittaker', app.Count_2, BaseVar{2,2}, BaseVar{3,2}, BaseVar{4,2});
Here, runProcessor will:

Clear the shared memory buffer.
Create shared memory segments from the provided input arguments.
Call the registered function (e.g., whittaker) with the shared memory data.
Read and return the results from the shared memory.
Custom Functions
Place your custom Python function files in the custom_functions directory. They will be automatically imported and registered as static methods within the function registry upon package initialization.

Data Handling & Conversion
Large Arrays:
The package is designed to handle large 1D NumPy arrays efficiently. When a function returns such an array, it is wrapped appropriately to prevent iterating over millions of elements unnecessarily.

Logical Conversions:
If an element in shared memory is a logical scalar (e.g., logical(0) or logical(1)), it will be converted to its integer equivalent (0 or 1) for compatibility, especially when interfacing with MATLAB.

Troubleshooting
Memory Allocation Errors in MATLAB:
If you encounter errors like "Requested array exceeds the maximum possible variable size," consider processing your data in smaller chunks or downsampling the data.

Data Format Issues:
Ensure that when shared memory is read, arrays are in the expected orientation (e.g., converting 1×N arrays to N×1 if necessary) to avoid downstream issues.

Contributing
Contributions are welcome! If you have improvements, bug fixes, or additional features, please fork the repository and submit a pull request.

License
[Include your license information here, e.g., MIT License, Apache License, etc.]