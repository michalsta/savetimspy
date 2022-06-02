%load_ext autoreload
%autoreload 2
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

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")
fragHeLa = _get_d("*3343.d")
unfragHeLa = _get_d("*3342.d")

# source = unfrag5P
source = unfragHeLa
target = project_folder/"tests"/"hprs"/source.name
# target = source/"hprs"

HPR_intervals = make_overlapping_HPR_mz_intervals(
    min_mz=300,
    max_mz=1500,
)
# HPR_intervals = make_overlapping_HPR_mz_intervals(
#     min_mz=400,
#     max_mz=426,
# )
hprs = HPRS(
    HPR_intervals=HPR_intervals,
    dia_run=DiaRun(source),
    verbose=True,
)
# for cycle, hpr_idx, framedataset in tqdm(hprs.iter_nonempty_aggregated_cycle_hpr_data()):
#     pass
# for cycle, hpr_idx, framedataset in hprs.iter_all_aggregated_cycle_hpr_data(verbose=True):
#     pass

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
)
hprs.dia_run._acceptable_max_frame
hprs.dia_run.Frames.query("MsMsType==0").Id

# OK, we have some index error.
hprs.dia_run.max_cycle
hprs.dia_run.DiaFrameMsMsInfo
hprs.dia_run._get_frames_raw()

list(hprs.dia_run.DiaFrameMsMsInfo.groupby("cycle"))


# the error stems from iterating over non-existing cycle-step pairs:
# easy fix: iterate over all of them.

feature_folders = [Run4DFFv4_12_1(hpr_d, verbose=True) for hpr_d in hpr_folders]
hpr_clusters = []
for hpr_idx, ff in enumerate(feature_folders):
    hpr = read_4DFF_to_df_physical_dims_only(ff)
    hpr = get_extents_vol_centers(hpr)
    hpr["experiment"] = hpr_idx
    hpr_clusters.append(hpr)
