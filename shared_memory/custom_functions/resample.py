# custom_functions/resample.py

import numpy as np
from scipy.signal import resample_poly

def resample(Count_2, num, denom, padding=10):
    """
    Resample a signal using polyphase filtering with boundary padding.

    Parameters:
    Count_2 (array): Input signal array.
    num (int): Upsampling factor.
    denom (int): Downsampling factor.
    padding (int, optional): Number of points to pad at each end (default is 10).

    Returns:
    array: Resampled signal with padding removed.
    """
    # Compute the mean of the first and last 10 points
    start_mean = np.mean(Count_2[:10])
    end_mean = np.mean(Count_2[-11:])

    # Create a padded signal using the computed boundary means
    padded_signal = np.concatenate([
        np.full((padding,), start_mean),
        Count_2,
        np.full((padding,), end_mean)
    ])

    # Perform resampling with specified up/down factors
    resampled_signal = resample_poly(padded_signal, up=num, down=denom)

    # Remove padding and return the final resampled signal
    return resampled_signal[padding:-padding]
