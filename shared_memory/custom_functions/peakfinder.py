# custom_functions/peakfinder.py
import numpy as np
from functools import wraps
from collections import defaultdict


## - - - - - Utilities - - - - - ##

def list_to_np_array(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        # Convert all list-type arguments to np.array
        new_args = [np.array(arg) if isinstance(arg, list) else arg for arg in args]
        new_kwargs = {k: np.array(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        
        return func(*new_args, **new_kwargs)
    
    return wrapper

@list_to_np_array
def convert_to_numpy(*args):
    # To convert list to arrays
    return args

def clean_float64_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the original function
        result = func(*args, **kwargs)
        
        # Convert the result to a NumPy array of dtype float64
        try:
            result = np.array(result, dtype=np.float64)
        except Exception as e:
            pass        
        return result
    return wrapper

## - - - - - Helper Functions - - - - - ##

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

## - - - - - Main Functions - - - - - ##

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
            
        p1, p2, p3, MX, MX_Loc = convert_to_numpy(p1, p2, p3, MX, MX_Loc)  # as such
    
        TEST = PEAKS_FWHM
        
        Area = np.array(AreaFWHM)
        
        WidthFWHM = (p3-p1) * T_res

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

@clean_float64_output
def PeaksBeta_V2(Count, PoisLamda, thresh, time_raw, Buff):

    s = np.array(Count)
    #s += np.random.rand(len(s)) * 1e-9
    Data = s
    
    if Buff['On'] == 1:
        g = np.array(s)
        g[g < PoisLamda] = 0
        g[g >= PoisLamda] = 1 
        
        for j in range(Buff['Numb']):
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
    ps = np.concatenate(([p[0]], np.diff(p, axis = 0)))   # Detect transitions in p
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
    time = np.array(time_raw)[ind_new]
    
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
    
    # Calculate the AUC for burst
    Area = (2 * SUM - first - last)/2
    
    TiMaxBurst = Output[:,1]        # Extract peak time
    PkMaxBurst = m[FndAboveThresh]      # Extract peak value

    TiLow = accumarray(ind_new3, time, (lambda x: x[0]))    # Start time of each burst
    TiHigh = accumarray(ind_new3, time, (lambda x: x[-1]))  # End time of each burst
        
    ind_start = np.where(np.atleast_1d(ind_start) == 1)[0] + Output[:,0]
    PeakIndex = ind_start+1

    return TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, TEST, PeakIndex

def peakfinder(CountBase, PoisLamda, thresh, time, T_res, FullWidthHM,
               WidthLimit, CurrentLimit, FileCondition, Buff):
    
    TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, Event_all, PeakIndex = \
        FindPeaks_V2(CountBase, PoisLamda, thresh, time, T_res, FullWidthHM, 
                     WidthLimit, CurrentLimit, FileCondition, Buff)

    return TiMaxBurst, PkMaxBurst, MeanBurst, TiLow, TiHigh, Area, Event_all, PeakIndex
