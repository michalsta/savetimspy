import numba
import numpy as np
import functools
import pathlib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import matplotlib.pyplot as plt
import zstd

from opentimspy import OpenTIMS


project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")
fragHeLa = _get_d("*3343.d")
unfragHeLa = _get_d("*3342.d")



# Getting a map scan (list index) -> number of peaks



op = OpenTIMS(unfragHeLa)
Frames = pd.DataFrame(op.frames)
total_scans = Frames.NumScans.max()
df = pd.DataFrame(op.query(frames=range(1,3), columns=("scan","tof","intensity")))
df = df.groupby(["scan","tof"], as_index=False, sort=True).intensity.sum()


scans = df.scan.to_numpy()
tofs = df.tof.to_numpy()
intensities = df.intensity.to_numpy()



peak_cnts = get_peak_cnts(total_scans, scans)
modify_tofs(tofs)
if not isinstance(intensities, np.ndarray) or not intensities.dtype == np.uint32:
    intensities = np.array(intensities, np.uint32)
interleaved = np_zip(tofs, intensities)
real_data = get_realdata(peak_cnts, interleaved)


compressed_data = zstd.ZSTD_compress(bytes(real_data), 1)


tdf_bin = open( '/tmp/analysis.tdf_bin', 'wb')
tdf_bin.write((len(compressed_data)+8).to_bytes(4, 'little', signed = False))
tdf_bin.write(int(total_scans).to_bytes(4, 'little', signed = False))
tdf_bin.write(compressed_data)

