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
)
from savetimspy.write_frame_datasets import write_frame_datasets
import shutil
import pathlib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import sqlite3
import opentimspy

from py4DFF import Run4DFFv4_12_1
from MSclusterparser.parser import read_4DFF_to_df_physical_dims_only
from dia_common import DiaRun

from savetimspy.write_frame_datasets import (
    write_frame_datasets,
    FrameDataset
)

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")

source = unfrag5P
target = project_folder/"tests"/"hprs"/source.name

HPR_intervals = make_overlapping_HPR_mz_intervals(
    min_mz=400,
    max_mz=426,
)

hprs = HPRS(
    HPR_intervals=HPR_intervals,
    dia_run=DiaRun(source),
    verbose=True,
)


# hprs.plot_scans_and_steps_used_by_hypothetical_precursor_range(hpr_idx=0)
hprs.step_to_scan_hpr_dfs

# lets try to rewrite the iterator
import numba
import numpy as np

cols_to_choose = ("scan","tof","intensity")
tof_intensity = ["tof","intensity"]
# cycle_step_tuples = zip(
#     hprs.dia_run.DiaFrameMsMsInfo.cycle, 
#     hprs.dia_run.DiaFrameMsMsInfo.step,
# )
cycle = 1


# get all MS2 peaks per cycle

ms2_frame_ids_per_cycle = hprs.dia_run.cycle_step_to_ms2_frame(
    cycle=cycle,
    step=np.arange(hprs.dia_run.min_step, hprs.dia_run.max_step+1)
)
raw_peaks_per_cycle = hprs.dia_run._get_frames_raw(
    frame_ids=ms2_frame_ids_per_cycle,
    columns=("frame","scan","tof","intensity"),
)
raw_peaks_per_cycle['step'] = hprs.dia_run.ms2_frame_to_step(raw_peaks_per_cycle["frame"])
del raw_peaks_per_cycle['frame']
raw_peaks_per_cycle = pd.DataFrame(raw_peaks_per_cycle)
raw_peaks_per_cycle = raw_peaks_per_cycle.set_index(["step", "scan"])

# need to make something like 
hprs.hpr_quadrupole_matches

# this one time:
step_to_scan_hpr_df = [ ]
for step, df in enumerate(hprs.step_to_scan_hpr_dfs):
    df = df.reset_index()
    df["step"] = step
    step_to_scan_hpr_df.append(df)
step_to_scan_hpr_df = pd.concat(step_to_scan_hpr_df, ignore_index=True)
step_to_scan_hpr_df = step_to_scan_hpr_df.set_index(["step", "scan"])


raw_peaks_prod = pd.merge(
    step_to_scan_hpr_df,
    raw_peaks_per_cycle,
    left_index=True,
    right_index=True,
)
# figure out what should be here
for hpr_idx, data in raw_peaks_prod.groupby("hpr_idx")[tof_intensity]:




# it = iter(hprs)
# hprs.dia_run.DiaFrameMsMsInfo
# hpr_byte = next(it)
# print(hpr_byte.cycle, hpr_byte.step, hpr_byte.hpr_idx)


# datasets = [[] for _ in range(len(hprs.HPR_intervals))]
# prev_cycle = 0
# for hpr_byte in hprs:

#     if prev_cycle < hpr_byte.cycle:
#         prev_cycle = hpr_byte.cycle
#         datasets = merge_data(datasets)
#         for dataset in datasets:

#         for _lst in hpr_2_step_data_per_cycle:
#             _lst.clear()


# it = hprs.full_iter()
# hpr_byte = next(it)
# len(hpr_byte.intensities)


# df = pd.DataFrame({
#     "scan":         hpr_byte.scans,
#     "tof":          hpr_byte.tofs,
#     "intensity":    hpr_byte.intensities
# })
# df


