%load_ext snakeviz
%load_ext autoreload
%autoreload 2
import collections
import functools
import itertools
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import opentimspy
import matplotlib.pyplot as plt
import numpy as np
import numba
import pathlib

from dia_common import DiaRun
from opentimspy.sql import table2dict
from py4DFF import Run4DFFv4_12_1
from savetimspy.get_hprs import *
from savetimspy.fs_ops import get_limits, set_soft_limit, print_limits 
from savetimspy.write_frame_datasets import FrameSaveBundle
from savetimspy.get_hprs import write_hprs
from tqdm import tqdm
from savetimspy.numba_helper import dedup_v2, deduplicate
from MSclusterparser.raw_peaks_4DFF_parser import (
    dump_4DFF_clustered_events_to_hdf,
    iter_pandas_hdf_groups,
    get_clustered_raw_events_count,
    get_RawPeaks_chunk,
    iter_pairs_of_indices,
    RawPeaksFrom4DFF_HDF5,
)
from MSclusterparser.parser import read_4DFF_to_df_physical_dims_only
from MSclusterparser.boxes_ops import get_extents_vol_centers
from kilograms import scatterplot_matrix

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P     = _get_d("*3516.d") 
unfrag5P   = _get_d("*3517.d")
fragHeLa   = _get_d("*3343.d")
unfragHeLa = _get_d("*3342.d")

source = unfragHeLa
# source = unfrag5P
HPR_intervals = make_overlapping_HPR_mz_intervals()
# HPR_intervals = HPR_intervals.loc[[100]]
paths = write_hprs(
    HPR_intervals=HPR_intervals,
    source=source,
    verbose=True
)

# hprs = HPRS(
#     HPR_intervals=HPR_intervals,
#     dia_run=DiaRun(fromwhat=source,
#                    preload_data=False,
#                    columns=("frame", "scan", "tof", "intensity"))
# )

# TransformHPR_100 = opentimspy.OpenTIMS(paths[0])
# TransformHPR_100_data = pd.DataFrame(TransformHPR_100.query(TransformHPR_100.frames['Id']))
# with plt.style.context('dark_background'):
#     scatterplot_matrix(TransformHPR_100_data[["retention_time","inv_ion_mobility","mz"]], weights=TransformHPR_100_data.intensity, show=False)
#     plt.suptitle("Intensity Weighted HPR 100: 900-912 m/z")
#     plt.show()

# hprs.plot_scans_and_steps_used_by_hypothetical_precursor_range(100)
hpr_to_boxes = {}
for hpr_idx, path in zip(HPR_intervals.index, paths):
    clusters_path = Run4DFFv4_12_1(path, verbose=True)
    clusters = read_4DFF_to_df_physical_dims_only(clusters_path)
    hpr_to_boxes[hpr_idx] = get_extents_vol_centers(clusters)


cols = ["retention_time", "inv_ion_mobility", "mz"]
extents = [f"{c}_extent" for c in cols]

filtered_clusters = clusters.query("retention_time_extent < 50 and mz_extent < .06")
with plt.style.context('dark_background'):
    scatterplot_matrix(
        filtered_clusters[extents],
        weights=filtered_clusters.intensity, show=False)
    plt.suptitle("Intensity Weighted HPR 100 Boxes Extents: 900-912 m/z")
    plt.show()


with plt.style.context('dark_background'):
    scatterplot_matrix(
        np.log10(clusters[extents]+.0001),
        weights=clusters.intensity, show=False)
    plt.suptitle("Intensity Weighted HPR 100 Boxes Extents: 900-912 m/z")
    plt.show()






# HPR_intervals = HPR_intervals.iloc[[34]]
# HPR_intervals = HPR_intervals.iloc[[100]]

# combine_steps_per_cycle=False
# verbose=True
# _max_iterations=200
# _max_iterations=None
# _soft_limit=4096
# compression_level=1
dia_run = DiaRun(
    fromwhat=source,
    preload_data=False,
    columns=("frame", "scan", "tof", "intensity"),
)
hprs = HPRS(
    HPR_intervals=HPR_intervals,
    dia_run=dia_run,
)

# it = hprs.iter_hpr_transform(progressbar=True)
# hpr_idx, cycle, diagonal, data = next(it)
# hpr_idx, cycle, diagonal
# # # initially slow because of the matrix computations
# for hpr_idx, cycle, diagonal, data in hprs.iter_hpr_transform(progressbar=True):
#     print(hpr_idx, cycle, diagonal)
#     if cycle==1:
#         break

for hpr_idx, cycle, diagonal, data in hprs.iter_hpr_transform(progressbar=True):
    print(diagonal)
    if cycle == 2:
        break
    assert is_sorted(data["scan"]), f"{hpr_idx=} {cycle=} {diagonal=}"

pd.DataFrame(hprs.dia_run.opentims.frames)


hprs.get_diagonals_to_frame_to_retention_time()

hprs.dia_run.Frames

hprs.dia_run.opentims.frames["Time"]

MS1_Ids = hprs.dia_run.opentims.frames["Id"]
MS1_Ids_plus_end = np.append(MS1_Ids, )

hprs.dia_run.DiaFrameMsMsInfo
MS1_retention_times = hprs.dia_run.opentims.frames["Time"][hprs.dia_run.opentims.frames["MsMsType"] == 0]


retention_times = hprs.dia_run.opentims.frames["Time"]
self = hprs

all_subdiagonals_retention_times = get_interpolated_retention_times(
    retention_times, len(self.dia_run.DiaFrameMsMsInfo)*len(self.diagonals))

frame_id + (diagonal-1)/len(diagonals)?
# That will be OK:

ids = hprs.dia_run.opentims.frames["Id"]
retention_times = hprs.dia_run.opentims.frames["Time"]
msms_types = hprs.dia_run.opentims.frames["MsMsType"]
base = ids[msms_types==0]
diag_to_retention_times = {}

# Just make it for all things till last step and that's it: add more on top
artificial_ids = np.concatenate(
    [base + 21*(diagonal-1)/len(self.diagonals) for diagonal in self.diagonals]
)

diagonal_to_frame_to_retention_time[1]

len(diagonal_to_frame_to_retention_time[diagonal])



diagonal_to_
all_subdiagonals_retention_times

xx = np.arange()



get_interpolated_retention_times(
    hprs.dia_run.opentims.frames["Time"],
    100
)
w = get_interpolated_retention_times(
    hprs.dia_run.opentims.frames["Time"],
    len(retention_times)//2
)
plt.hist(np.diff(w), bins=100, alpha=.5)
plt.hist(np.diff(hprs.dia_run.opentims.frames["Time"][::2]), bins=100, alpha=.5)
plt.show()


rt_diffs = np.diff(hprs.dia_run.opentims.frames['Time'])
from clusterMS.stats import discrete_histogram
rt_diffs = discrete_histogram(collections.Counter(rt_diffs))

plt.scatter(rt_diffs.index, rt_diffs)
plt.hist(rt_diffs, bins=100)
plt.show()
hprs.plot_scans_and_steps_used_by_hypothetical_precursor_range(100)



# from kilograms import scatterplot_matrix


# with plt.style.context('dark_background'):
#     scatterplot_matrix(pd.DataFrame(diagonal_3))

# is_sorted(np.lexsort([data["tof"], data["scan"]]))
# for i, group in enumerate(groups_tags):
#     diagonal_to_datasets[group].append()

