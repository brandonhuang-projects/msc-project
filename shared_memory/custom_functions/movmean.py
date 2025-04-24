# custom_functions/movmean.py
import numpy as np
from numba import njit

@njit
def movmean_jit(A, k):
    # Assume A is already a NumPy array of floats
    n = A.shape[0]
    
    # If window size is 1, return a copy of the original array
    if k == 1:
        return A.copy()
    
    # Compute cumulative sum with an inserted leading zero
    cumsum = np.empty(n + 1, dtype=A.dtype)
    cumsum[0] = 0.0
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + A[i]
    
    # Compute the core moving mean for full windows
    mlen = n - k + 1
    M_core = np.empty(mlen, dtype=A.dtype)
    for i in range(mlen):
        M_core[i] = (cumsum[i + k] - cumsum[i]) / k
    
    # Determine boundary sizes
    f = k // 2              # Number of front boundary points
    e = (k - 1) // 2        # Number of end boundary points
    
    # Prepare output array
    out = np.empty(n, dtype=A.dtype)
    
    # Front boundary: growing window from start
    for i in range(f):
        s = 0.0
        for j in range(i + 1):
            s += A[j]
        out[i] = s / (i + 1)
    
    # Core: centered full windows
    for i in range(mlen):
        out[f + i] = M_core[i]
    
    # End boundary: shrinking window to end
    for i in range(e):
        s = 0.0
        for j in range(n - (i + 2), n):
            s += A[j]
        out[f + mlen + i] = s / (i + 2)
    
    return out


def movmean(Count_2, Time_2, MovMean_Const):
    
    Base = movmean_jit(Count_2, MovMean_Const)

    Base = Base - (0.75 * Base.std(ddof=1))
    
    CountBase = Count_2 - Base
    
    # Set Negative values to 0
    CountBase = np.maximum(CountBase, 0)
    
    return CountBase