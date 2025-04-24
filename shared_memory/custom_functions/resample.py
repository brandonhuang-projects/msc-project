# custom_functions/resamples.py
import numpy as np
from scipy.signal import resample_poly

def resample(Count_2, num, denom, padding=10):
    # Compute start and end means
    start_mean = np.mean(Count_2[:10])
    end_mean = np.mean(Count_2[-11:])
    
    # Create padded signal
    padded_signal = np.concatenate([
        np.full((padding,), start_mean),
        Count_2,
        np.full((padding,), end_mean)
    ])
    
    # Resample and remove padding
    resampled_signal = resample_poly(padded_signal, up=num, down=denom)
    return resampled_signal[padding:-padding]
