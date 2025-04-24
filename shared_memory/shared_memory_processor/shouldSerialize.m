function flag = shouldSerialize(value)
% should_serialize determines whether the given value should be JSON encoded.
%
% For numeric, char, or string values, no serialization is needed.
% For cell arrays, it attempts to convert them to a matrix.
%   - If conversion succeeds (i.e. the cell array is uniform), then serialization is not needed.
%   - If conversion fails, the cell array is non-uniform and should be serialized.
% For structs (or other types), serialization is required.
%
% Usage:
%   flag = should_serialize(value)
%
% Returns:
%   flag - true if the value should be JSON encoded, false otherwise.

    if isnumeric(value) || ischar(value) || isstring(value)
        flag = false;
    elseif iscell(value)
        try
            % Attempt to convert the cell array to a matrix.
            cell2mat(value);
            flag = false;
        catch
            flag = true;
        end
    elseif isstruct(value)
        flag = true;
    else
        flag = true;
    end
end
