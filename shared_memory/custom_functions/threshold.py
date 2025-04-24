# custom_functions/threshold.py
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson
from scipy.special import gamma
from math import sqrt
from numba import njit

@njit
def fast_histogram(data, bins, step):
    hist = np.zeros(len(bins), dtype=np.int64) # No len(bins)-1 since MATLAB includes
    
    for value in data:
        bin_index = int(value / step)   # Convert int for index
        if 0 <= bin_index < len(hist):
            hist[bin_index] += 1
            
    return hist

# Poission PMF
def Pois(u, x):
    return (u**x) * np.exp(-u) / gamma(x + 1)

def threshold(CountBase, STD, SO, ThrStep):
    
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

    # Compute fitted Poisson PMF
    PoisFit = poisson.pmf(np.arange(1, len(StepVec) + 1), PoisLamda)

    # Normalize to histogram scale
    PoisFit = PoisFit / (np.max(PoisFit) * np.max(Hist))

    thresh = PoisLamda + (STD * sqrt(PoisLamda))
    thresh = thresh * Step
    PoisLamda = PoisLamda * Step * SO
    
    return thresh, PoisLamda, StepVec, Hist, PoisFit
