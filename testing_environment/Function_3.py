
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:48:35 2024

@author: brand

    
"""

# Peak Finding -------------------------------------------

import numpy as np
from functools import wraps
from collections import defaultdict

import Function_2

def list_to_np_array(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        # Convert all list-type arguments to np.array
        new_args = [np.array(arg) if isinstance(arg, list) else arg for arg in args]
        new_kwargs = {k: np.array(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        
        return func(*new_args, **new_kwargs)
    
    return wrapper

@list_to_np_array
def buffer(*args):
    # To convert list to arrays
    return args


def FindPeaks_V2(CountBase, PoisLamda, thresh,
                 time, T_res, FullWidthHM, WidthLimit,
                 CurrentLimit, FileCondition, Buff):
    
    # Call PeaksBeta_V2
    TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, TEST, PeakIndex = \
        PeaksBeta_V2(CountBase, PoisLamda, thresh, time, Buff)
    
    # Filter out invalid bursts based on conditions
    ValZero = (TiHigh - TiLow) > 0
    TiMaxBurst = TiMaxBurst[ValZero]
    PkMaxBurst = PkMaxBurst[ValZero]
    MeanBurst = MeanBurst[ValZero]
    TiLow = TiLow[ValZero]
    TiHigh = TiHigh[ValZero]
    Area = Area[ValZero]
    TEST = TEST[ValZero] #[row for row, include in zip(TEST, ValZero) if include]
    PeakIndex = PeakIndex[ValZero]
    
    if WidthLimit['Low_on'] == 1:
        ValC = (TiHigh - TiLow) >=  WidthLimit['Low_with']
        TiMaxBurst = TiMaxBurst(ValC);
        PkMaxBurst = PkMaxBurst(ValC);
        MeanBurst = MeanBurst(ValC);
        TiLow = TiLow(ValC);
        TiHigh = TiHigh(ValC);
        Area = Area(ValC);
        TEST = TEST[ValZero]
        PeakIndex = PeakIndex(ValC);


    if WidthLimit['Upper_on'] == 1:
        ValC = (TiHigh - TiLow) <=  WidthLimit['Upper_with']
        TiMaxBurst = TiMaxBurst(ValC);
        PkMaxBurst = PkMaxBurst(ValC);
        MeanBurst = MeanBurst(ValC);
        TiLow = TiLow(ValC);
        TiHigh = TiHigh(ValC);
        Area = Area(ValC);  
        TEST = TEST[ValZero]
        PeakIndex = PeakIndex(ValC);
    
    
    if CurrentLimit['on'] == 1:
        # PkMaxBurst
        ValD = (PkMaxBurst <= CurrentLimit['Value']/1000) # convert to nA
        TiMaxBurst = TiMaxBurst(ValD)
        PkMaxBurst = PkMaxBurst(ValD)
        MeanBurst = MeanBurst(ValD)
        TiLow = TiLow(ValD)
        TiHigh = TiHigh(ValD)
        Area = Area(ValD)
        TEST = TEST[ValZero]
        PeakIndex = PeakIndex(ValD)

    PeakIndexRaw = np.round(TiMaxBurst/T_res)
    
    
    if FullWidthHM['On'] ==  1:        # full width half max of bursts
        # For each burst:

        MX = []             # Max value
        MX_Loc = []         # Index of max value 
        Y = []              # Half-maximum threshold 
        p1 = []             # Left boundary index of FWHM 
        p2 = []             # Relative right boundary index 
        p3 = []             # Absolute right boundary index  (for whole burst)
        
        PEAKS_FWHM = []     # Burst segments
        AreaFWHM = []       # AUC (sum)
        
        # Iterating over bursts
        for X in TEST:
            max_value = np.max(X)
            max_index = np.argmax(X)
            
            MX.append(max_value)         # Append max value to MX list
            MX_Loc.append(max_index)     # Append index of max value to MX_Loc list
            
            # Step 2: Calculate the half-maximum threshold
            half_max = max_value / FullWidthHM['factor']
            Y.append(half_max)
            
            left_index = np.argmin(abs(X[:max_index + 1] - half_max))
            p1.append(left_index)
            
            right_index = np.argmin(abs(X[max_index:] - half_max))
            p2.append(right_index)
            
            absolute_right_index = min(right_index + max_index + 1, len(X) - 1)
            p3.append(absolute_right_index)
            
            fwhm_segment = X[left_index:absolute_right_index + 1]
            PEAKS_FWHM.append(fwhm_segment)
            
            auc = (2 * np.sum(fwhm_segment) - fwhm_segment[0] - fwhm_segment[-1]) / 2
            AreaFWHM.append(auc)
            
        p1, p2, p3, MX, MX_Loc = buffer(p1, p2, p3, MX, MX_Loc)  # as such
    
        TEST = PEAKS_FWHM
        
        Area = AreaFWHM
        
        WidthFWHM = (p3-p1)*T_res
        
        TiLowFWHM = TiMaxBurst + ((p1 - MX_Loc) * T_res)
        TiHighFWHM = TiMaxBurst + ((p3 - MX_Loc) * T_res)
        
        TiLow = TiLowFWHM
        TiHigh = TiHighFWHM
        
        PeakIndex = (PeakIndex - MX_Loc + (p3 - p1)/2+1);
    
    if WidthLimit['Low_on'] == 1:
        ValC = (TiHigh - TiLow) >= WidthLimit['Low_width']
        TiMaxBurst = TiMaxBurst(ValC)
        PkMaxBurst = PkMaxBurst(ValC)
        MeanBurst = MeanBurst(ValC)
        TiLow = TiLow(ValC)
        TiHigh = TiHigh(ValC)
        Area = Area(ValC)
        TEST = TEST[ValZero]
        PeakIndex = PeakIndex(ValC)

    return TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, TEST, PeakIndex 

def PeaksBeta_V2(Count, PoisLamda, thresh, time, Buff):
    
    def strfind(arr,subarray):
        for i in range(len(arr) - len(subarray) + 1):
            if arr[i : i + len(subarray)] == subarray:
                return i
        return -1             
    
    def accumarray(ind, data, func = np.sum):
        # !! ASSUMES IND IS SORTED and 1-based indexed!!
        
        ind = np.array(ind) - 1 # Convert 1-based indexing to 0-based
        
        sz = np.max(ind) + 1 # Define size based on the maximum index
        
        # Sum data by index
        if func == np.sum:
            
            # np.bincount for efficient summation based on index array
            return np.bincount(ind, weights=data, minlength=sz)
        
        # Calculate mean by index
        elif func == np.mean:
            
            # Uses np.bincount to get sum and count per index
            sum_array = np.bincount(ind, weights=data, minlength=sz)
            count_array = np.bincount(ind, minlength=sz)
            
            # Divides to find the mean; zero-handling via pre-initialized array
            # and subsequent restriction
            return np.divide(sum_array, count_array, 
                             out=np.zeros_like(sum_array, dtype=float), 
                             where=(count_array > 0))
        
        # Calculate max value by index
        # O(n) complexity
        elif func == np.max:
            
            # Initialize with lowest index
            output = np.full(sz, 0.0)   
            
            for idx, value in zip(ind, data):
                
                # Update if current value is greater
                if value > output[idx]:         
                    output[idx] = value
            
            return output 
        
        # Retrieve first item by index
        # O(n) complexity
        elif func == (lambda x: x[0]):
            output = np.zeros(sz)       # Initialize output
            current_index = -1          # Track index
            
            # First occurrence only
            for idx, value in zip(ind, data):   
                if idx != current_index:        
                    output[idx] = value
                    current_index = idx

            return output
         
        # Retrieve last item by index
        # O(n) complexity
        elif func == (lambda x: x[-1]):
            output = np.zeros(sz)       # Initialize output
            
            # Overwriting so the last occurrence is kept
            for idx, value in zip(ind, data): 
                output[idx] = value  

            return output
        
        # Apply other functions on accumulated lists
        else:
            output = [None] * sz        # Initialize list output
            accumulator = defaultdict(list)     # Store data by index via defaultdict
            
            # Accumulate data into lists for each index
            for i, idx in enumerate(ind):
                accumulator[idx].append(data[i])
        
            # Apply function on accumulated lists for each index
            for idx, values in accumulator.items():
                    output[idx] = func(values)
                
            try:
                # Convert to numpy array; replace None with 0 
                return np.array([x if x is not None else 0 for x in output])
            except:
                try:
                    # Return object array if invalid
                    return np.array([x if x is not None else 0 for x in output], dtype=object)
               
                except Exception as e:
                    # Return as list
                    return [x if x is not None else 0 for x in output]

    @list_to_np_array
    def findTime(indx, s2, time):
    
    # !! Relies on 0-based indexing of indx
    # Replicates purpose, not data fom MATLAB original
        
        subset = s2[indx]
        ix_loc = np.argmax(subset)
        
        ix = [ix_loc + 1, time[indx[ix_loc]]]

        return ix
    
    # - - - - - Function Starts  - - - - - 
    
    s = np.array(Count)
    #s += np.random.rand(len(s)) * 1e-9
    Data = s
    
    if Buff['On'] == 1:
        g = np.array(s)
        g[g < PoisLamda] = 0
        g[g >= PoisLamda] = 1        
        
        for j in range(Buff['Num']):
            r = strfind(g,[1,1,1] + [0]*j + [1])
            for n in range(j):
                g[r+3+n-2] = 1 # Adjust for 0-based indexing
                
            r = strfind([1] + [0]*j + [1,1,1])
            for n in range(j):
                g[r+1+n-2] = 1 # Adjust for 0-based indexing
                
        s[g == 0] = 0 # Zero elements in s where g is 0
        
    else:
        s[s < PoisLamda] = 0 # Zero elements below PoisLamda
        
    p = (s != 0).astype(int) # 1 if s is non-zero else 0
    
    # Detect and transform bursts in p
    ps = np.concatenate(([p[0]], np.diff(p)))   # Detect transitions in p
    ps[ps != 1] = 0                             # Marks transition
    ps = np.cumsum(ps) * p                      # Unique labels for each sequence
    
    # Keep non-zero values
    s = s[s != 0] 
    #g = g[g != 0]
    p = ps[ps != 0]
    
    m = accumarray(p, s, func = np.max)
    
    # - - - - - - - - - - 
    
    FndAboveThresh = np.where(np.atleast_1d(m) >= thresh)[0] # atleast_1d() for consistency
    ind_new = np.isin(ps, FndAboveThresh + 1) # !! Accommodate for 1-indexing of ps
    
    s2 = Data[ind_new]
    time = time[ind_new]
    
    # Identify start points of new bursts
    ind_new2 = np.concatenate(([ind_new[0]], np.diff(ind_new.astype(int))))
    ind_new2 = np.where(ind_new2 != 1, 0, ind_new2)
        
    ind_start = ind_new2    # Save start markers
    
    # Cumulative sums of burst indicators
    ind_new2 = np.cumsum(ind_new2) * ind_new
    ind_new3 = ind_new2[ind_new2 != 0]
    
    first = accumarray(ind_new3, s2, (lambda x: x[0]))  # First item in each burst
    last = accumarray(ind_new3, s2, (lambda x: x[-1]))  # Last item in each burst
    MeanBurst = accumarray(ind_new3, s2, np.mean)       # Mean value for each burst
    SUM = accumarray(ind_new3, s2, np.sum)              # Total sum for each burst

    # Collect burst segment as an array
    TEST = accumarray(ind_new3, s2, (lambda x: np.array(x))) # SLOW; no need for [x] call

    # Find peak time for each burst segment
    Output = accumarray(ind_new3, range(len(s2)), (lambda x: findTime(x, s2, time)))
    # Second arg adjusted for 0-based indexing
    
    try:
        Output = np.array(Output)
    except:
        pass
    
    __ = ''''first = Output[:,2]
    last = Output[:,3];
    MeanBurst = Output[:,4];
    SUM = Output[:,5]'''
    
    # Calculate the AUC for burst
    Area = np.array((2 * SUM - first - last)/2)
    
    TiMaxBurst = Output[:,1]        # Extract peak time
    PkMaxBurst = m[FndAboveThresh]      # Extract peak value

    TiLow = accumarray(ind_new3, time, (lambda x: x[0]))    # Start time of each burst
    TiHigh = accumarray(ind_new3, time, (lambda x: x[-1]))  # End time of each burst
        
    
    ind_start = np.where(np.atleast_1d(ind_start) == 1)[0] + Output[:,0]
    PeakIndex = ind_start

    
    return TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, TEST, PeakIndex

def pre_function(load_from_matlab = True):

    kwargs = Function_2.pre_function()
    kwargs = Function_2.function(**kwargs)    
    
    # - - - - - 
    path = r'C:\Users\brand\OneDrive\UNI\Chem\Year 5\MSc\Scripts\data'
    
    if load_from_matlab:
        Count_3 = np.loadtxt(path+'\\'+'Count_3.csv', delimiter=',')
        Time_3 = np.loadtxt(path+'\\'+'Time_3.csv', delimiter=',')
        
    else:
        # Load ABF file
        kwargs['abf'].setSweep(sweepNumber=0, channel=1)
        
        # Extract and return sweep data
        Count_3 = kwargs['abf'].sweepY.astype(np.float64)
        Time_3 = kwargs['abf'].sweepX.astype(np.float64)*10**5 # Account for scaling on MATLAB 
    
    ResampleValue = kwargs['abf'].dataRate
    
    FileCondition = False
    
    FWHM_CheckBox = True;

    FWHMDropDown = '2';
    
    LWidthLimit_CheckBox = 0; # == 1 dependance 
    
    LWidthLimit = 0.0005;
    
    UWidthLimit_CheckBox = 0; # == 1 dependance 
    
    UWidthLimit = 0.01;
    
    UCurrentLimit_CheckBox = 0; # == 1 dependance 
    
    CurrentBelowVal = 50;
    
    Buffer_CheckBox = 0;
    
    BufferBinsNo = 5;
            
    BaselineLengthIndividual = 75;
    
    BaselineLength = 100; 
    
    #kwargs['thresh'] = 2.718718654741767e+02
    
    kwargs['PoisLamda'] = 1.397561117796946e+02
    
    return {**kwargs, **locals()} # Return local variables
    
def function(CountBase, PoisLamda, thresh, Time_2, Time_3, Count_2, Count_3, 
             FileCondition, ResampleValue, FWHM_CheckBox, FWHMDropDown, 
             LWidthLimit_CheckBox, LWidthLimit, UWidthLimit_CheckBox, UWidthLimit, 
             UCurrentLimit_CheckBox, CurrentBelowVal, BaselineLength,
             Buffer_CheckBox, BufferBinsNo, timer=False, **kwargs):

    # Define time resolution
    T_res = ResampleValue
    T_res = T_res*1e-6;
    
    time = Time_2
    
    FullWidthHM = {'On': FWHM_CheckBox,
                   'factor': float(FWHMDropDown)}
     
    WidthLimit = {'Low_on': LWidthLimit_CheckBox,
                  'Low_width': LWidthLimit,
                  'Upper_on': UWidthLimit_CheckBox,
                  'Upper_width': UWidthLimit}
    
    CurrentLimit = {'on': UCurrentLimit_CheckBox,
                  'Value': CurrentBelowVal}
    
    Buff = {'On': Buffer_CheckBox,
            'Numb': BufferBinsNo}
    
    # Call FindPeaks_V2
    TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, Event_all, PeakIndex = \
        FindPeaks_V2(CountBase, PoisLamda, thresh, time, T_res, FullWidthHM, 
                     WidthLimit, CurrentLimit, FileCondition, Buff)
    
    if True:
        #plot
        pass


    MXX = np.max(CountBase)
    
    y = MXX * np.ones(len(TiMaxBurst))
    
    y2 = MXX * np.ones(len(TiMaxBurst))
    
    Time_2, Time_3 = buffer(Time_2, Time_3) # Convert to numpy arrays
    
    # Check if Time_2 and Time_3 match
    if  np.array_equal(Time_3, Time_2):
        Count_temp = Count_2
        CountBaseLoc = CountBase
    
    else:
        ia = np.isin(Time_3, Time_2)  # Boolean array of matching elements in 
    
        # Create dictionary for Time_2 indexing (1-based)
        time_2_dict = {value: idx + 1 for idx, value in enumerate(Time_2)}
        ib = np.array([time_2_dict.get(elem, 0) for elem in Time_3])
    
        # Initialize arrays
        Count_temp = np.zeros(len(Time_3))
        CountBaseLoc = Count_temp
        
        # Assign values where times match
        Count_temp[ia] = Count_2[ia]
        CountBaseLoc[ia] = CountBase[ia]
    
    r = '''return TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, Event_all, PeakIndex, Count_temp, CountBaseLoc,\
        CountBase, PoisLamda, thresh, time, T_res, FullWidthHM, \
                     WidthLimit, CurrentLimit, FileCondition, Buff'''

    Event = []      # Initialized MATLAB's structure mimic

    for j in range(len(TiLow)):
        
        # Terminals for reference window
        CusumReferencedStartPoint = round(TiLow[j]/T_res) - BaselineLength
        if CusumReferencedStartPoint < 1:
            CusumReferencedStartPoint = 1
            
        CusumReferencedEndPoint = round(TiHigh[j]/T_res) - BaselineLength
        if CusumReferencedEndPoint > len(Time_3):
            CusumReferencedEndPoint = len(Time_3)
        
        # Define start and end point for events
        StartPoint = round(TiLow[j]/T_res)
        EndPoint = round(TiHigh[j]/T_res)
        
        MnBkGMax = 20;
        if MnBkGMax > BaselineLength:
            MnBkGMax = BaselineLength - 1
        
        # Subtract background mean
        background_mean = np.mean(CountBaseLoc[CusumReferencedStartPoint:CusumReferencedStartPoint+ MnBkGMax])
        PkMaxBurst[j] = PkMaxBurst[j] - np.nan_to_num(background_mean)
        if PkMaxBurst[j] <= 0:
            PkMaxBurst[j] = 1e-6
            
        # Initialize event dictionary to append to parent Event structure    
        event = {}
        
        event["CountsExtract"] = Count_temp[CusumReferencedStartPoint:CusumReferencedEndPoint]
        event["TimeExtract"] = Time_3[CusumReferencedStartPoint:CusumReferencedEndPoint]
        event["Fit"] = np.full(len(event["TimeExtract"]), np.nan)
        event["TimeEvent"] = Time_3[StartPoint:EndPoint]
        event["NumberOfLevels"] = 0
        event["TiLow"] = TiLow[j]
        event["TiHigh"] = TiHigh[j]
        event["PkMaxBurst"] = PkMaxBurst[j]
        event["Area"] = Area[j]
        event["TiMaxBurst"] = TiMaxBurst[j]
        event["MeanBurst"] = MeanBurst[j]
        event["Event_all"] = Count_3[StartPoint:EndPoint]
        event["Event_Base"] = event["Event_all"]
        event["Event_Base_full"] = CountBaseLoc[CusumReferencedStartPoint:CusumReferencedEndPoint]
    
        Event.append(event) 

        # Set peak threshold and step for histogram binning
        PkThresh = np.nanmax(PkMaxBurst)
        PkStep = PkThresh/50
        PkVec = np.arange(0, PkThresh + PkStep, PkStep)
    
        N, edges = np.histogram(Event[j]['Event_Base_full'], bins=PkVec)
        
        event['IndCurrentHist.Edges']  = edges[:-1]   
        event['IndCurrentHist.CurrentHist'] = N

    return {**kwargs, **locals()} # Return local variables

def post_function():
    pass


# LINE 94, 105, 114

# Elapsed time (MATLAB): 0.417282 s ± 0.022676 s (mean ± std. dev. of 100 runs)
# Elapsed time (Python): 1.917527 s ± 0.076491 s (mean ± std. dev. of 100 runs)

# With optimized rand
# Elapsed time (Python): 0.399956 s ± 0.020163 s (mean ± std. dev. of 100 runs)

# With optimized accumarray()
# Elapsed time (Python): 0.387645 s ± 0.019910 s (mean ± std. dev. of 100 runs)

# Completed Function
# Elapsed time (Python): 0.414728 s ± 0.025601 s (mean ± std. dev. of 100 runs)


# Completed Function
# Elapsed time (Python): 0.442453 s ± 0.014450 s (mean ± std. dev. of 100 runs)
# Elapsed time (MATLAB): 1.251004 s ± 0.242985 s (mean ± std. dev. of 100 runs)

# No random:
# Elapsed time (Python): 0.361533 s ± 0.029395 s (mean ± std. dev. of 100 runs)

# Other optimizations (clean run):
# Elapsed time (Python): 0.322304 s ± 0.012701 s (mean ± std. dev. of 30 runs)

# Elapsed time (Python): 0.317502 s ± 0.012443 s (mean ± std. dev. of 100 runs)

# Elapsed time (Python): 0.307654 s ± 0.013741 s (mean ± std. dev. of 100 runs)

# float32 vs float64
# Elapsed time (Python): 0.313334 s ± 0.012945 s (mean ± std. dev. of 500 runs)
# Elapsed time (Python): 0.314660 s ± 0.011953 s (mean ± std. dev. of 100 runs)
