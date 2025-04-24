# validate_outputs_py.m
# MATLAB wrapper to call validator.py
function validate_outputs_py(matlabFile, pythonFile)
    if nargin < 2
        matlabFile = 'output_matlab.mat';
        pythonFile = 'output_python.mat';
    end
    command = sprintf('python validator.py "%s" "%s"', matlabFile, pythonFile);
    status = system(command);
    if status ~= 0
        warning('Validator script did not execute successfully.');
    end
end