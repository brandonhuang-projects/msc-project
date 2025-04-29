# custom_functions/whittaker.py

from pybaselines import Baseline

def whittaker(y, lam, p, order):
    """
    Apply Whittaker smoothing using asymmetric least squares baseline correction.

    Parameters:
    y (array): Input signal.
    lam (float): Smoothing parameter.
    p (float): Asymmetry parameter.
    order (int): Order of difference penalty.

    Returns:
    array: Estimated baseline.
    """
    # Perform asymmetric least squares baseline correction
    Base, _ = Baseline().asls(y, lam=lam, p=p, diff_order=order, max_iter=10)
    return Base