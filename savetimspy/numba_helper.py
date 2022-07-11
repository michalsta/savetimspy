import cmath
import numba
import numpy as np
import numpy.typing as npt
from collections import Counter

# do not use this: dedup_v2 is 10 times faster.
#@jit(nopython=True) #TODO: make this work
def deduplicate(scans, tofs, intensities):
    # It is not easy to specify the type of the Counter.
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


# this really works faster
@numba.jit
def binary_search(scans, min_scan, max_scan):
    return np.searchsorted(scans, (min_scan, max_scan) )


@numba.jit
def linear_search(scans, min_scan, max_scan):
    for i in range(len(scans)):
        if scans[i] >= min_scan:
            break
        else:
            i+= 1
    min_idx = i
    for j in range(min_idx, len(scans)):
        if scans[j] >= min_scan:
            break
        else:
            j+= 1
    max_idx = j
    return min_idx, max_idx


@numba.jit(nopython=True)
def dedup_sorted(xx, yy, weights, order):
    if len(xx) <= 1:# nothing to do and code below needs len(xx) > 1
        return (xx, yy, weights)
    x_prev = xx[order[0]]
    y_prev = yy[order[0]]
    xx_res = []
    yy_res = []
    ww_res = []
    w_agg = weights[order[0]]
    for i in range(1, len(xx)):    
        x = xx[order[i]]
        y = yy[order[i]]
        if x > x_prev or y > y_prev:
            xx_res.append(x_prev)
            yy_res.append(y_prev)
            ww_res.append(w_agg)
            w_agg = 0
            x_prev = x
            y_prev = y
        w_agg += weights[order[i]]
    if x == x_prev and y == y_prev:
        xx_res.append(x)
        yy_res.append(y)
        ww_res.append(w_agg)
    return (
        np.array(xx_res, dtype=xx.dtype),
        np.array(yy_res, dtype=yy.dtype),
        np.array(ww_res, dtype=weights.dtype)
    )

# 10 times faster than deduplicate above.
def dedup_v2(xx, yy, weights):
    order = np.lexsort([yy, xx])
    return dedup_sorted(xx, yy, weights, order)


@numba.njit
def get_group_tags_starts_ends(groups):
    if len(groups) == 1:
        return [(groups[0],0,1)]
    res = []
    i_prev = 0
    tag_prev = groups[i_prev]
    for i, tag in enumerate(groups):
        if tag != tag_prev:
            res.append((tag_prev, i_prev, i))
            i_prev = i
            tag_prev = tag
    if i_prev < i:
        res.append((tag, i_prev, len(groups)))
    return res
