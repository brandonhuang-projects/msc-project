function metadata = createSharedMemorySegments(memoryDir, varargin)
% createSharedMemorySegments Create multiple shared memory segments in the "memory" folder
%   metadata = createSharedMemorySegments(memoryDir, data1, data2, ...)
%
%   - memoryDir: The directory where all shared memory identifier files and
%                the metadata JSON file will be stored (e.g., a folder named "memory").
%   - varargin: One or more data arrays (of supported MATLAB native types)
%               that you wish to transfer.
%
% This function loops over each data array, generates a unique shared memory
% identifier, explicitly creates a file (storing the identifier) in the memoryDir,
% sets the shared memory path to that file, and calls set_shared_memory_data() to store the data.
% It then collects metadata for each segment and writes a JSON file named "metadata.json"
% in memoryDir.
%
% Returns:
%   metadata: A MATLAB struct containing the metadata for all segments.

    % Ensure the memory directory exists; create it if needed.
    if ~exist(memoryDir, 'dir')
        mkdir(memoryDir);
    end

    % Metadata file path
    metaFilePath = fullfile(memoryDir, '_metadata.json');
    
    % Load existing metadata if it exists
    if exist(metaFilePath, 'file')
        fid = fopen(metaFilePath, 'r');
        rawJson = fread(fid, '*char')';
        fclose(fid);
        metadata = jsondecode(rawJson);
    else
        error("Metadata file not found: %s. Ensure Python has initialized the session.", metaFilePath);
    end

    % Keep session ID unchanged
    sessionId = metadata.session_id;

    % Prepare a structure to hold metadata for all segments.
    metadata.segments = {};

    % Loop over each input data array.
    for i = 1:numel(varargin)

        data = varargin{i};
        varId = lower(dec2hex(randi([0, 2^32-1], 1),8));
        
        segMeta = struct();
        
        if isnumeric(data) && isscalar(data) && isreal(data)
            segMeta.varId = varId;
            segMeta.source = 'MATLAB';
            segMeta.inline = true;
            segMeta.inlineData = data;
            segMeta.dataType = class(data);
            segMeta.dimensions = [];
            segMeta.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
            
        else   
            segmentFile = fullfile(memoryDir, [varId,'.shmem']);
            set_shared_memory_path(segmentFile);
    
            % For complex types, JSON-encode the data.
            if shouldSerialize(data)
                data = jsonencode(data);
                segDataType = 'json';
                segDimensions = [];
            else
                segDataType = class(data);
                segDimensions = size(data);
            end
            
            % Write the data to the shared memory segment.
            set_shared_memory_data(data);
            
            % Collect metadata for this segment.
            segMeta.varId        = varId;
            segMeta.source       = 'MATLAB';
            segMeta.filePath     = segmentFile;
            segMeta.dataType     = segDataType;
            segMeta.dimensions   = segDimensions;
            segMeta.timestamp    = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
        end
        
        % Append the segment metadata.
        metadata.segments{end+1} = segMeta;

    end

    % Write the JSON string to a metadata file inside the memory directory.
    metadata.session_id = sessionId;
    fid = fopen(metaFilePath, 'w');
    fprintf(fid, '%s', jsonencode(metadata, 'PrettyPrint', true));
    fclose(fid);
    
    % Display a message.
    % fprintf('Created %d shared memory segments.\nMetadata stored in:\n%s\n', ...
    %    numel(varargin), metaFilePath);
end
