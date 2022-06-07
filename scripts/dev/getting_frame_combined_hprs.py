%load_ext autoreload
%autoreload 2
%load_ext snakeviz
from savetimspy.get_frames import write_frames
from savetimspy.get_ms1 import write_ms1
from savetimspy.get_ms2 import write_ms2, combined_ms2_frames_generator
from savetimspy.get_diagonals import write_diagonals
from savetimspy.get_hprs import (
    write_hprs,
    make_overlapping_HPR_mz_intervals,
    HPRS,
    HPRbyte,
    write_hprs,
)
from savetimspy.write_frame_datasets import write_frame_datasets
import shutil
import pathlib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import sqlite3
import opentimspy

from dia_common import DiaRun
from MSclusterparser.parser import read_4DFF_to_df_physical_dims_only
from py4DFF import Run4DFFv4_12_1
from savetimspy.write_frame_datasets import (
    write_frame_datasets,
    FrameDataset
)
from tqdm import tqdm
from MSclusterparser.boxes_ops import *
import matplotlib.pyplot as plt
import plotnine as p
import typing

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")
fragHeLa = _get_d("*3343.d")
unfragHeLa = _get_d("*3342.d")

# source = unfrag5P
# source = unfragHeLa
source = fragHeLa
target = source/"hprs_faster"

HPR_intervals = make_overlapping_HPR_mz_intervals(
    min_mz=300.0,
    max_mz=1_500.0,
)

# %%snakeviz
hpr_folders = write_hprs(
    HPR_intervals=HPR_intervals,
    source=source,
    target=target,
    combine_steps_per_cycle=True,
    min_coverage_fraction=0.5,# no unit
    window_width=36.0,# Da
    min_scan=1,
    max_scan=None,
    compression_level=1,
    make_all_frames_seem_unfragmented=True,
    verbose=True,
    # _max_iterations=10_000,
)

feature_folders = [Run4DFFv4_12_1(hpr_d, verbose=True) for hpr_d in hpr_folders]
points = [len(opentimspy.OpenTIMS(hpr_folder)) for hpr_folder in hpr_folders]
HPR_intervals['center'] = HPR_intervals.eval("(hpr_start + hpr_stop)/2.0")
plt.plot(HPR_intervals.center, points)
plt.scatter(HPR_intervals.center, points)
plt.xlabel("Center of the HPR")
plt.ylabel("raw events count")
plt.show()



hpr_clusters = []
for hpr_idx, ff in tqdm(enumerate(feature_folders)):
    hpr = read_4DFF_to_df_physical_dims_only(ff)
    hpr = get_extents_vol_centers(hpr)
    hpr["experiment"] = hpr_idx
    hpr_clusters.append(hpr)

extent_cols = ["mz_extent","inv_ion_mobility_extent","retention_time_extent"]
pd.plotting.scatter_matrix(
    hpr_clusters[np.argmax(points)].query("retention_time_extent < 55")[extent_cols],
    hist_kwds={"bins":101},
    grid=True,
    s=1
)
plt.suptitle(f"Feature Extents For Most Event-Rich ({max(points):_} events) HPR No {np.argmax(points)}.")
plt.show()

pd.plotting.scatter_matrix(
    hpr_clusters[np.argmin(points)].query("retention_time_extent < 55")[extent_cols],
    hist_kwds={"bins":101},
    grid=True,
    s=1
)
plt.suptitle(f"Feature Extents For Least Event-Rich ({min(points):_} events) HPR No {np.argmin(points)}.")
plt.show()

hpr_clusters_df = pd.concat(hpr_clusters, ignore_index=True)
hpr_clusters_df.query("vol>0")
MS1 = read_4DFF_to_df_physical_dims_only(source/f"{source.name}.features")

hpr_clusters_df.to_hdf(source/"features.hdf", "hprs")
MS1.to_hdf(source/"features.hdf", "ms1")


