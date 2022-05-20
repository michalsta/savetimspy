from numba import jit
#from numba.typed import Dict
import numpy as np
import numpy.typing as npt
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


@jit(cache=True)
def get_groups_as_consecutive_ints(
    xx: np.array,
    dtype=np.uint32,
    start=1,
):
    yy = np.empty(len(xx), dtype)
    x_prev = xx[0]
    y = start
    i = 0
    for x in xx:
        if x != x_prev:
            x_prev = x
            y += 1
        yy[i] = y
        i += 1
    return yy


@jit
def coordinatewise_range(
    starts: npt.NDArray[int],
    ends:   npt.NDArray[int]
) -> npt.NDArray[int]:
    """Spread compact scans into long lists.
    
    For instance:
         ScanNumBegin  ScanNumEnd
                    0           2
                    1           4
                    5           7
                    0           3

    Would result in:
        np.array([  0,1,  1,2,3,  5,6,  0,1,2  ])

    
    """
    res = []
    for start, stop in zip(starts, ends):
        for i in range(start, stop):
            res.append(i)
    return np.array(res)
