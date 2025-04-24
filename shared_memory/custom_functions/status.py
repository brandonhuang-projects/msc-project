# custom_functions/status.py
import numpy as np

def status(data1, data2, data3, data4, data5, data6, data7, data8, data9):
    
    out1 = data1.upper() if isinstance(data1, str) else data1
    out2 = np.mean(data2)
    out3 = np.sum(data3)
    out4 = np.max(data4)
    out5 = np.sum(data5)
    out6 = np.conjugate(data6)
    out7 = np.round(data7,2)
    out8 = [data8[:2], data8[2:]]
    out9 = {
        'a': data9['a'] + 1,
        'b': data9['b'].upper(),
        'c': float(np.sum(data9['c'])),
        'd': not data9['d']
        }
    
    return out1, out2, out3, out4, out5, out6, out7, out8, out9