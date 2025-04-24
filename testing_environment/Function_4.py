# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:57:25 2024

@author: brand
"""
import numpy as np
import matplotlib.pyplot as plt
import pyabf


# pip install pentapy
# pip install pybaselines
from pybaselines import Baseline

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
    
    BaseVar = {'Step': 3, 'Lambda': 10000000, 'p': 0.01, 'Order': 2}
    
    y = Count_2
    lam = BaseVar['Lambda']
    p = BaseVar['p']
    step = BaseVar['Step']
    order = BaseVar['Order']
    
    return locals()

def function(y, lam, p, step, order, **kwargs):
    
    Base, _ = Baseline().asls(y, lam = lam, p = p, diff_order = order, max_iter = 10)
    
    return Base

def post_function():
    plt.close('all')

# Elapsed time (MATLAB): 10.601881 s ± 0.155521 s (mean ± std. dev. of 30 runs)
# Elapsed time (Python): 5.205663 s ± 0.046945 s (mean ± std. dev. of 30 runs)