function varargout = readSharedMemorySegments(bufferPath)
% readSharedMemoryMetadata Reads shared memory metadata and retrieves the data.
%   dataMap = readSharedMemoryMetadata(metaPath) reads the JSON metadata file
%   at metaPath (which should contain a field 'segments', an array of metadata
%   structs for each shared memory segment). For each segment, if the varName
%   ends with '.shmem', the prefix is removed. Then, set_shared_memory_path is called
%   with the filePath, and get_shared_memory_data is called to retrieve the data.
%
%   The returned dataMap is a containers.Map object with keys as variable names (without
%   the '.shmem' suffix) and values as the retrieved data.
%
%   Example:
%       metaPath = fullfile(memoryDir, 'metadata.sjson');
%       dataMap = readSharedMemoryMetadata(metaPath);
%       % To access data for variable 'data1' (originally stored as 'data1.shmem'):
%       myData = dataMap('data1');
%
    metaPath = fullfile(bufferPath, '_metadata.json');

    % Check that the metadata file exists.
    if exist(metaPath, 'file') ~= 2
        error('Metadata file not found: %s', metaPath);
    end

    % Read the JSON file.
    fid = fopen(metaPath, 'rt'); % Open in text mode
    if fid == -1
        error('Could not open metadata file: %s', metaPath);
    end
    jsonStr = fread(fid, '*char')';
    fclose(fid);
    
    metadata = jsondecode(jsonStr);
    if ~isfield(metadata, 'segments')
        error('Metadata file is missing the "segments" field.');
    end
    
    segments = metadata.segments;
    
    % Handle empty metadata case
    if isempty(segments)
        varargout = {};
        return;
    end

    % Prepare a cell array to hold the outputs.
    dataList = cell(numel(segments), 1);
    c = 0;
    
    % Loop over each segment.
    for i = 1:numel(segments)
        if iscell(segments)
            seg = segments{i};
        else
            seg = segments(i);
        end

        if isfield(seg, 'source') && strcmp(seg.source, 'MATLAB')
            continue; % Skip this iteration and move to the next one
        end

        % Use inline data if available.
        if isfield(seg, 'inline') && seg.inline

            c = c + 1;
            dataList{c} = seg.inlineData;
            continue;
        end
        
        % Retrieve variable name and file path.
        filePath = seg.filePath;

        % Set the shared memory path for this segment.
        set_shared_memory_path(filePath);
        
        % Retrieve the data.
        data = get_shared_memory_data();
        
        % If the data was stored as JSON, decode it.
        if isfield(seg, 'dataType') && strcmp(seg.dataType, 'json')
            data = jsondecode(data);
        end

        % Store the data in the map using the variable name as the key.
        c = c + 1;
        dataList{c} = data;
    end
    
    % Preallocate varargout to the actual number of valid entries.
    varargout = cell(1, c);
    for j = 1:length(dataList)
        varargout{j} = dataList{j};
    end                                                                                             
end
