# write_output.py
# Utility to write output to a .mat file
from scipy.io import savemat

def write_output(data, filename="output_python.mat"):
    """
    Save result dictionary or ndarray to .mat format for MATLAB comparison
    """
    if isinstance(data, dict):
        savemat(filename, data)
    else:
        savemat(filename, {"result": data})
