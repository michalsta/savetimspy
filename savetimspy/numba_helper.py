#from numba import jit
#from numba.typed import Dict
import numpy as np
from collections import Counter


#@jit(nopython=True) #TODO: make this work
def deduplicate(scans, tofs, intensities):
    C = Counter()
    for i in range(len(scans)):
        C[(scans[i], tofs[i])] += intensities[i]
    
    ret_scans = np.empty(len(C), dtype=np.uint32)
    ret_tofs = np.empty(len(C), dtype=np.uint32)
    ret_intensities = np.empty(len(C), dtype=np.uint32)

    idx = 0
    for (scan, tof), intens in sorted(C.items()):
        ret_scans[idx] = scan
        ret_tofs[idx] = tof
        ret_intensities[idx] = intens
        idx += 1

    return ret_scans, ret_tofs, ret_intensities
