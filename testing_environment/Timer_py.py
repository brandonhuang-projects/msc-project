# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:38:35 2024

@author: brand
"""
import time as time_import
import statistics

import pyabf # ABF File Handling
    #“Analysis of electrophysiological recordings was performed with custom software written for this project using Python 3.10 and the pyABF package¹.”
    #[1] Harden, SW (2022). pyABF 2.3.5. [Online]. Available: https://pypi.org/project/pyabf


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

from line_profiler import LineProfiler


print("Imports Initialized")

# ----- Timer Functions -------------------------

global_kwargs = None

def time_python(pre_function, function, post_function, repeat):
    """
    Measures and profiles the execution time of a function over a specified number of repetitions.
    
    This function executes a `pre_function` to perform any setup tasks, then repeatedly times 
    the execution of the `function` with any additional arguments provided by `pre_function`.
    A `post_function` is executed after each iteration to handle any cleanup tasks. The first 
    run is excluded from timing to account for overhead from the pre-function.
    
    Args:
        pre_function (callable): A setup function that prepares global keyword arguments 
                                 and other initial states for `function`.
        function (callable): The primary function whose execution time will be measured.
        post_function (callable): A cleanup function executed after each call to `function`.
        repeat (int): Number of times to repeat the execution of `function` for timing.
    
    Returns:
        None. Outputs timing results (mean and standard deviation) to the console.
    
    Prints:
        - Timing for each run (excluding the first run).
        - Average elapsed time and standard deviation across repetitions.
    """
    
    global global_kwargs
    
    time_list = [] # Store execution times
    result = None # Output placeholder
    
    # Pre-script setup
    print('-- Running pre-setup script...')
    
    if not global_kwargs:
        print("-- Importing...")
        global_kwargs = pre_function()
        global_kwargs['timer'] = True
    
    # Per designated repeats:
    for i in range(0, repeat + 1):
        print(f'-- Timing main function (Run {i} of {repeat})', flush=True)
        start_time = time_import.time()     # Start timer
        result = function(**global_kwargs)         # Run main function
        end_time = time_import.time()       # End timer
        
        if i != 0:       # Flush first repeat due to pre_function overhead
                        # causing overestimations in runtime
            time_list.append(end_time - start_time) 
            print(f'Elapsed time is {round(time_list[-1],6)} seconds.')
            
        else:
            print(f'Elapsed time flushed for first run')
        post_function()
        
    # Mean and standard deviation
    mean_time = statistics.mean(time_list)
    std_dev_time = statistics.stdev(time_list)

    print(f"Elapsed time (Python): {mean_time:.6f} s ± {std_dev_time:.6f} s") #(mean ± std. dev. of {repeat} runs)")


def run_line_profile(function,**kwargs):
    """
    Profiles the line-by-line execution time of a given function.
    
    This function uses the `LineProfiler` to measure the time taken for each line of code
    within the specified `function`. Additional arguments for the function are passed via `kwargs`.
    
    Args:
        function (callable): The function to be profiled.
        **kwargs: Arbitrary keyword arguments passed to the function being profiled.
    
    Returns:
        None. Outputs the profiling statistics to the console.
    """
    kwargs['timer'] = True
    lp = LineProfiler()
    lp_wrapper = lp(function)
    lp_wrapper(**kwargs)
    return lp.print_stats()

# ----- Pre-load -------------------------

if True:
    
    import Function_1 as test_module

    time_python(test_module.pre_function,
                test_module.function,
                test_module.post_function,30)
    
    #run_line_profile(test_module.function, **test_module.pre_function())

