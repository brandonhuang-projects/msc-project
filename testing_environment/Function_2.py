# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:33:19 2024

@author: brand
"""

# Thresholding and Poisson distribution -------------------------------------------

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson
from scipy.special import gamma
import pyabf
from math import sqrt
from numba import njit

import Function_1
from Function_1 import movmean

import time

@njit
def fast_histogram(data, bins, step):
    hist = np.zeros(len(bins), dtype=np.int64) # No len(bins)-1 since MATLAB includes
    
    for value in data:
        bin_index = int(value / step)   # Convert int for index
        if 0 <= bin_index < len(hist):
            hist[bin_index] += 1
            
    return hist

def pre_function(load_from_matlab = True): 

    kwargs = Function_1.pre_function(load_from_matlab) # Pre-process function arguments
    kwargs = Function_1.function(**kwargs) # Run main function with arguments
    
    # Variables from AppEditor
    ThresholdVariables = {'std': 7,
                          'SO': 1.8, 
                          'Overide': False,
                          'Overide Step': 0.01,
                          'Overide Max': 0.5}
    
    # Override settings
    ThrStep = {'Overide': ThresholdVariables['Overide'],
               'Step': ThresholdVariables['Overide Step'],
               'MX': ThresholdVariables['Overide Max']}
    
    STD = ThresholdVariables['std']
    SO = ThresholdVariables['SO']
    
    return {**kwargs, **locals()} # Return local variables

def function(CountBase, STD, SO, ThrStep, timer=False, **kwargs):
    # Calculate the first difference
    Df = np.diff(CountBase)
    # Replace any NaN values 0
    Df[np.isnan(Df)] = 0
    
    # Compute mean of the absolute deltas
    Step = np.mean(abs(Df))
    
    # Set minimum step value
    if Step < 0.0001:
        Step = 0.001
        
    max_val = np.max(CountBase)
        
    StepVec = np.arange(0,max_val/2,Step)

    # Refine StepVec if too small
    if len(StepVec) < 25:
        Step = Step/2
        StepVec = np.arange(0,max_val/2,Step)
    
    # Override Threshold
    if ThrStep['Overide'] == True:
        Step = ThrStep['Step']
        MX = ThrStep['MX']
        StepVec = range(0,MX,Step)

    # CountBase histogram with StepVec bins    
    Hist = fast_histogram(CountBase, StepVec, Step)

    # Set curve fit upper bound; inf if small
    if len(CountBase) > 1000:
        UL = (np.mean(CountBase[0:1000])/Step)*10
    else:
        UL = float('inf')
        
    # Normalize
    Hist = Hist/np.sum(Hist) # MATLAB includes 0 at end after histc()
    
    # Remove first element
    Hist = Hist[1:]
    StepVec = StepVec[1:]
    
    # Poission PMF
    def Pois(u, x):
        return (u**x) * np.exp(-u) / gamma(x + 1)
    
    pois_array = np.arange(1, len(StepVec) + 1)
    
    # Least squares curve fitting | X ~ Pois(λ)
    # (2) Adapted from https://uk.mathworks.com/help/optim/ug/lsqcurvefit.html#buuhcjo-3
    try:
        # least-squares with discrete integer indieces
        result = least_squares(lambda u: Pois(u, pois_array) - Hist,
                               x0 = 5, bounds = (0, UL), method = 'trf',
                               ftol=1e-6, xtol=1e-6, gtol=1e-6, diff_step=1e-6) # as per (2)
                               #max_nfev = (100 * len(StepVec)), ) # as per (2)
    except:
        result = least_squares(lambda u: Pois(u, pois_array) - Hist,
                               x0 = 2, bounds = (0, UL), method = 'trf',
                               ftol=1e-6, xtol=1e-6, gtol=1e-6, diff_step=1e-6) # as per (2)
                               #max_nfev = (100 * len(StepVec)), ) # as per (2)
    PoisLamda = result.x[0]
    PoisLamda = 7.814518319593709       # Explicit definition for consistency
                                        # Delete when not debugging
    
    # Compute fitted Poisson PMF
    PoisFit = poisson.pmf(np.arange(1, len(StepVec) + 1), PoisLamda)
    
    # Normalize to histogram scale
    PoisFit = PoisFit / (np.max(PoisFit) * np.max(Hist))
    
    thresh = PoisLamda + (STD * sqrt(PoisLamda))
    thresh = thresh * Step
    PoisLamda = PoisLamda * Step * SO
    
    # Plot
   
        
    return {**kwargs, **locals()} # Return local variablesnbn
        
def post_function():
    pass
    
# Elapsed time (Python): 1.262698 s ± 0.017612 s (mean ± std. dev. of 100 runs)
# Elapsed time (MATLAB): 0.164687 s ± 0.062861 s (mean ± std. dev. of 100 runs)

# Some issues with conherence between MATLAB's lsqcurvefit() and Python's least_squares()
# can't seem to get it closer
# ** PoisLamda = 7.809590809363397 (Python)
# ** PoisLamda = 7.805923454240931 (MATLAB) 

# Optimizations
# Elapsed time (Python): 0.169957 s ± 0.014801 s (mean ± std. dev. of 100 runs)


# Elapsed time (Python): 0.074149 s ± 0.003938 s (mean ± std. dev. of 100 runs)
