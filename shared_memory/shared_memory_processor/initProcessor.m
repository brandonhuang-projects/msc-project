function func_names = initProcessor()
    
    global SharedMemoryProcessor

    SharedMemoryProcessor = struct;
    
    SharedMemoryProcessor.MATLAB_PATH = fileparts(mfilename('fullpath'));

    SharedMemoryProcessor.DIR= fileparts(fileparts(mfilename('fullpath')));
    
    SharedMemoryProcessor.WINPYTHON_PATH = fullfile(SharedMemoryProcessor.DIR, 'WPy64-31190b5', 'python-3.11.9.amd64', 'python.exe'); 
    SharedMemoryProcessor.pe = pyenv('Version', SharedMemoryProcessor.WINPYTHON_PATH);
    
    SharedMemoryProcessor.SHARED_MEMORY_PATH = fullfile(SharedMemoryProcessor.DIR, 'shared_memory-main');
    addpath(SharedMemoryProcessor.SHARED_MEMORY_PATH);
    addpath(fullfile(SharedMemoryProcessor.SHARED_MEMORY_PATH, 'matlab'));
    SharedMemoryProcessor.BUFFER_PATH = fullfile(SharedMemoryProcessor.DIR, 'shared_memory_processor', '__ipc_buffer__');
    
    % SharedMemoryProcessor.PYTHON_EXE = pe.Executable;
    % SharedMemoryProcessor.PYTHON_SCRIPT = fullfile(SharedMemoryProcessor.SHARED_MEMORY_PATH, 'shared_memory_processor.py');
    
    %%%
    
    insert(py.sys.path, int32(0), SharedMemoryProcessor.DIR);
    SharedMemoryProcessor.module = py.importlib.import_module('shared_memory_processor');
    
    SharedMemoryProcessor.functions = py.shared_memory_processor.Functions().get_function_list;

    func_names = string(py.list(SharedMemoryProcessor.functions.keys()));

    SharedMemoryProcessor.processor = py.shared_memory_processor.SHMEM_Processor();