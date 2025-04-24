# custom_functions/whittaker.py
from pybaselines import Baseline

def whittaker(y, lam, p, order):
    
    Base, _ = Baseline().asls(y, lam = lam, p = p, diff_order = order, max_iter = 10)
    return Base
