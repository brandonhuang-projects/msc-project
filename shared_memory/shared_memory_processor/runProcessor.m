function varargout = runProcessor(func, varargin)
    
    global SharedMemoryProcessor

    SharedMemoryProcessor.processor.clear_vars();

    % Initialize cell arrays for names and data
    tic
    
    for k = 1:numel(varargin)
        if islogical(varargin{k}) && isequal(size(varargin{k}), [1,1])
            varargin{k} = double(varargin{k});
        end
    end

    SharedMemoryProcessor.metadata = createSharedMemorySegments(SharedMemoryProcessor.BUFFER_PATH, varargin{:});

    if ischar(func) || isstring(func)
        func = SharedMemoryProcessor.functions{char(func)};
    end

    SharedMemoryProcessor.processor.call(func)

    [varargout{1:nargout}] = readSharedMemorySegments(SharedMemoryProcessor.BUFFER_PATH);

    for k = 1:nargout
        if isrow(varargout{k}) && numel(varargout{k}) > 1
            if ~(isstring(varargout{k}) || ischar(varargout{k}))
                varargout{k} = varargout{k}.';
            end
        end
    end
        
    toc
    