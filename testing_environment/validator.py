# validator.py
from scipy.io import loadmat
import numpy as np
import sys


def validate_outputs(matlab_path, python_path, atol=1e-8, rtol=1e-5):
    mat = loadmat(matlab_path)
    py = loadmat(python_path)

    keys = set(mat.keys()) & set(py.keys()) - {'__header__', '__version__', '__globals__'}
    if not keys:
        print("[WARN] No shared keys found between MATLAB and Python outputs.")
        return

    print("\n=== Fidelity Validation Report ===")
    for key in sorted(keys):
        arr1 = np.atleast_1d(mat[key])
        arr2 = np.atleast_1d(py[key])

        if arr1.shape != arr2.shape:
            print(f"[FAIL] {key}: Shape mismatch {arr1.shape} vs {arr2.shape}")
            continue

        abs_diff = np.abs(arr1 - arr2)
        rmse = np.sqrt(np.mean((arr1 - arr2) ** 2))
        max_diff = np.max(abs_diff)

        close = np.allclose(arr1, arr2, atol=atol, rtol=rtol)
        status = "PASS" if close else "FAIL"
        print(f"[{status}] {key}: max_diff={max_diff:.3e}, RMSE={rmse:.3e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validator.py output_matlab.mat output_python.mat")
    else:
        validate_outputs(sys.argv[1], sys.argv[2])
