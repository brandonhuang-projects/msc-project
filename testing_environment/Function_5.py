# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:48:22 2024

@author: brand
"""

import numpy as np
import pyabf
from numba import njit

@njit
def resample_padding(signal, num, denom, padding=10):
    # Resampling factor
    factor = num / denom

    # Padding using pre-computed means
    start_mean = np.sum(signal[:padding]) / padding  # Faster mean calculation
    end_mean = np.sum(signal[-padding:]) / padding
    padded_length = len(signal) + 2 * padding
    padded_signal = np.empty(padded_length)

    # Direct assignment for padding
    padded_signal[:padding] = start_mean
    padded_signal[padding:-padding] = signal
    padded_signal[-padding:] = end_mean

    if num == denom:  # No resampling needed
        resampled_signal = padded_signal

    else:
        # Resampling logic (linear interpolation)
        new_length = int(len(padded_signal) // factor)
        resampled_signal = np.zeros(new_length)  # Initialize output
        
        for i in range(new_length):
            # Indices for upsampling/downsampling
            original_idx = i * factor if denom > num else i / factor  
            low = int(original_idx)
            high = low + 1
            
            if high >= len(padded_signal):  # Boundary case
                resampled_signal[i] = padded_signal[low]
            else:
                frac = original_idx - low
                resampled_signal[i] = padded_signal[low] * (1 - frac) + padded_signal[high] * frac
   
    # Remove padding
    output_signal = resampled_signal[padding:-padding]
    
    return output_signal



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
    
    T_Resample = 7
    
    T_res = 10
    
    num = 10
    
    denom = 7
    
    return locals()

def function(Count_2, num, denom, padding=10, **kwargs):

    return resample_padding(Count_2, num, denom, padding=10)

def post_function():
    pass

# Elapsed time (MATLAB): 0.239498 s ± 0.054873 s (mean ± std. dev. of 30 runs)

# Elapsed time (Python v1): 0.119079 s ± 0.002727 s (mean ± std. dev. of 30 runs)
# Elapsed time (Python v2): 0.136744 s ± 0.005552 s (mean ± std. dev. of 30 runs)
# Elapsed time (Python vP): 0.139010 s ± 0.006082 s (mean ± std. dev. of 30 runs)