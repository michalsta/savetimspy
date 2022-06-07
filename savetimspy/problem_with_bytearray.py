%load_ext autoreload
%autoreload 2
import numpy as np
import numba

@numba.jit(forceobj=True)
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

x = np.arange(100000)
y = np.arange(100000)
%%timeit
z = get_realdata(x, y)

