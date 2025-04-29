# custom_functions/movmean.py
import numpy as np
from numba import njit

@njit
def movmean_jit(A, k):
    """
    Compute a moving average with boundary correction.

    Applies a centered moving mean with window size `k` to a 1D NumPy array `A`.
    Front and end boundaries use progressively smaller windows.
    
    Parameters
    ----------
    A : np.ndarray
        1D array of floats.
    k : int
        Window size for the moving mean.
    
    Returns
    -------
    np.ndarray
        Smoothed array of the same length as `A`.
    """
    n = A.shape[0]

    # If window size is 1, return a copy of the input array
    if k == 1:
        return A.copy()
    
    # Compute cumulative sum with a prepended zero
    cumsum = np.empty(n + 1, dtype=A.dtype)
    cumsum[0] = 0.0
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + A[i]
    
    # Compute moving mean for fully available windows
    mlen = n - k + 1
    M_core = np.empty(mlen, dtype=A.dtype)
    for i in range(mlen):
        M_core[i] = (cumsum[i + k] - cumsum[i]) / k
    
    # Boundary window sizes
    f = k // 2  # Points at the start
    e = (k - 1) // 2  # Points at the end
    
    # Prepare output array
    out = np.empty(n, dtype=A.dtype)
    
    # Front boundary: growing window
    for i in range(f):
        s = 0.0
        for j in range(i + 1):
            s += A[j]
        out[i] = s / (i + 1)
    
    # Core: centered full windows
    for i in range(mlen):
        out[f + i] = M_core[i]
    
    # End boundary: shrinking window
    for i in range(e):
        s = 0.0
        for j in range(n - (i + 2), n):
            s += A[j]
        out[f + mlen + i] = s / (i + 2)
    
    return out


def movmean(Count_2, MovMean_Const):
    """
    Apply baseline correction to a signal using moving mean smoothing.

    Smooths the input `Count_2` using a moving mean of width `MovMean_Const`,
    lowers the baseline by 0.75 standard deviations, and removes negative values.

    Parameters
    ----------
    Count_2 : np.ndarray
        1D array of counts or signal values.
    MovMean_Const : int
        Window size for the moving mean smoothing.
    
    Returns
    -------
    np.ndarray
        Baseline-corrected and thresholded signal.
    """
    Base = movmean_jit(Count_2, MovMean_Const)

    # Lower baseline by 0.75 times standard deviation
    Base = Base - (0.75 * Base.std(ddof=1))
    
    # Subtract baseline from original signal
    CountBase = Count_2 - Base
    
    # Set negative values to zero
    CountBase = np.maximum(CountBase, 0)
    
    return CountBase
