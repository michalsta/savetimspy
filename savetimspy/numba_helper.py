import numba
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


@numba.jit(cache=True)
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


@numba.jit
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


@numba.jit(cache=True)
def get_peak_cnts(total_scans, scans):
    peak_cnts = [total_scans]
    ii = 0
    for scan_id in range(1, total_scans):
        counter = 0
        while ii < len(scans) and scans[ii] < scan_id:
            ii += 1
            counter += 1
        peak_cnts.append(counter*2)
    peak_cnts = np.array(peak_cnts, np.uint32)
    return peak_cnts


@numba.jit(cache=True)
def modify_tofs(tofs, scans):
    last_tof = -1
    last_scan = 0
    for ii in range(len(tofs)):
        if last_scan != scans[ii]:
            last_tof = -1
            last_scan = scans[ii]
        val = tofs[ii]
        tofs[ii] = val - last_tof
        last_tof = val


@numba.jit
def np_zip(xx, yy):
    res = np.empty(2*len(xx), dtype=xx.dtype)
    i = 0
    for x,y in zip(xx,yy):
        res[i] = x
        i += 1
        res[i] = y
        i += 1
    return res


@numba.jit
def get_realdata(peak_cnts, interleaved):
    back_data = peak_cnts.tobytes() + interleaved.tobytes()
    real_data = bytearray(len(back_data))
    reminder = 0
    bd_idx = 0
    for rd_idx in range(len(back_data)):
        if bd_idx >= len(back_data):
            reminder += 1
            bd_idx = reminder
        real_data[rd_idx] = back_data[bd_idx]
        bd_idx += 4
    return real_data
