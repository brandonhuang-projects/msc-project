# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:33:43 2024

@author: brand
"""

# Baseline tracking -------------------------------------------

import numpy as np
import pyabf
from numba import njit

def movmean(A, k):
    # *Adapted from https://uk.mathworks.com/help/matlab/ref/movmean.html
    
    # Convert the input to a numpy array
    A = np.asarray(A)

    # Return array if k = 1
    if k == 1:
        return A

    cumsum = np.cumsum(np.insert(A, 0, 0))

    # Compute the moving mean using slicing
    M = (cumsum[k:] - cumsum[:-k]) /  k

    # Handle boundaries
    M_front = [np.mean(A[:i+1]) for i in range(k // 2)]
    M_end = [np.mean(A[-(i+2):]) for i in range((k-1)// 2)]
    
    return np.concatenate([M_front, M, M_end])

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


def pre_function(load_from_matlab = True):
    path = r'C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts\data'
    
    # Load ABF file
    abf = pyabf.ABF(r'C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts\test.abf')

    if load_from_matlab:
        Count_2 = np.loadtxt(path+'\\'+'Count_2.csv', delimiter=',')
        Time_2 = np.loadtxt(path+'\\'+'Time_2.csv', delimiter=',')
        
    else:
        abf.setSweep(sweepNumber=0, channel=0)
        
        # Extract and return sweep data
        Count_2 = abf.sweepY.astype(np.float64)
        Time_2 = abf.sweepX.astype(np.float64)*10**5 # Account for scaling on MATLAB 
    
    MovMean_Const = 10
    
    return locals()

def function(Count_2, Time_2, MovMean_Const, **kwargs):

    Base = movmean_jit(Count_2, MovMean_Const)

    Base = Base - (0.75 * Base.std(ddof=1))
    
    CountBase = Count_2 - Base
    
    # Set Negative values to 0
    CountBase = np.maximum(CountBase, 0)
    
    #fig, ax = plt.subplots()
    
    #TrackBaselinePlot = ax.plot(Time_2, CountBase)
    
    #ax.set_ylabel('Current (nA)')
    #ax.set_xlabel('Time (s)')
    
    return {**kwargs, **locals()} # Return local variables
    
def post_function():
    return

# With Graph
# Elapsed time (MATLAB): 0.084135 s ± 0.012103 s per loop (mean ± std. dev. of 100 runs)
# Elapsed time (Python): 0.492559 s ± 0.089067 s per loop (mean ± std. dev. of 100 runs)

# Without Graph
# Elapsed time (MATLAB): 0.070279 s ± 0.003211 s per loop (mean ± std. dev. of 100 runs)
# Elapsed time (Python): 0.185642 s ± 0.007264 s per loop (mean ± std. dev. of 100 runs)

# With new movmean
# Elapsed time (MATLAB): 0.089233 s ± 0.007313 s (mean ± std. dev. of 100 runs)
# Elapsed time (Python): 0.972058 s ± 0.014032 s (mean ± std. dev. of 100 runs)

# with newer
# Elapsed time (Python): 0.307510 s ± 0.028084 s (mean ± std. dev. of 30 runs)