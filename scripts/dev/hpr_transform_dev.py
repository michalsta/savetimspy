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
# hprs.plot_scans_and_steps_used_by_hypothetical_precursor_range(100)
self = hprs
hpr_indices = self._parse_input_hpr_indices(None)


hprs.get_hpr_cycle_step_data(100, 100, 19, columns=("scan","mz","intensity"))

def assert_data_the_same(old, new, columns=("scan","tof","intensity")):
    for col in columns:
        assert np.all(old[col] == new[col]), f"{col} differ"

for (hpr_idx_new, cycle_new, step_new, data_new), (hpr_idx_old, cycle_old, step_old, data_old) in zip(hprs.iter_hprs_v2(progressbar=True), hprs.iter_hprs()):
    assert hpr_idx_new==hpr_idx_old
    assert cycle_new==cycle_old
    assert step_new==step_old
    assert_data_the_same(data_old, data_new)
    

# @numba.njit
# def get_group_tags_starts_ends(groups):
#     # if len(groups) == 0:
#     #     return ([],[],[])
#     if len(groups) == 1:
#         return ([groups[0]],[0],[1])
#     i_prev = 0
#     starts = [0]
#     ends = []
#     tags = [groups[0]]
#     group_prev = groups[0]
#     for i, group in enumerate(groups):
#         if group != group_prev:
#             starts.append(i)
#             ends.append(i)
#             tags.append(group)
#             i_prev = i
#         group_prev = group
#     if i_prev < i:
#         ends.append(i)
#     return (tags, starts, ends)

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


# diagonal_to_scans = collections.defaultdict(list)
# diagonal_to_tofs = collections.defaultdict(list)
# diagonal_to_intensities = collections.defaultdict(list)


def empty_data(size, columns=("scan","tof","intensity"), dtype=np.uint32):
    return {col: np.empty(shape=(size,), dtype=dtype) for col in columns}


progressbar = True
# optimize this
subframes = {hpr_idx: hprs.get_subframe_df(hpr_idx) for hpr_idx in hprs.HPR_intervals.index}

hprs.plot_scans_and_steps_used_by_hypothetical_precursor_range(100)
# too slow for all that w do!
def hpr_transform(progressbar=False):
    hprs_times_steps_count = len(hpr_indices)*(self.dia_run.max_step+1)
    hpr_idx_step_diagonal_col_to_data = {}
    for i, (hpr_idx, cycle, step, data) in enumerate(self.iter_hprs(hpr_indices, progressbar)):
        groups = subframes[hpr_idx][data["scan"]-1, step]
        for diagonal, group_start, group_end in get_group_tags_starts_ends(groups):
            assert diagonal != 0, f"Found a point that does not belong to HPR={hpr_idx}"
            for col, values in data.items():
                hpr_idx_step_diagonal_col_to_data[hpr_idx, step, diagonal, col] = values[group_start:group_end]
        if (i + 1) % hprs_times_steps_count == 0:
            yield cycle, hpr_idx_step_diagonal_col_to_data
            hpr_idx_step_diagonal_col_to_data = {}
            # for diagonal, datasets in diagonal_to_data.items():
            #     event_cnts = [len(dataset['scan']) for dataset in datasets]
            #     data = empty_data(size=sum(event_cnts))
            #     j = 0
            #     # for event_cnt, dataset in zip(event_cnts, datasets):
            #     #     for col, array in data.items():
            #     #         array[ j:j+event_cnt ] = dataset[col]
            #     #     j += event_cnt
            #     # yield hpr_idx, cycle, diagonal, data
            #     yield hpr_idx, cycle, diagonal, diagonal_to_data
            # diagonal_to_data = collections.defaultdict(list)
it = hpr_transform()
next(it)




for _ in hpr_transform(progressbar=True):
    pass




for hpr_idx, cycle, step, data in hprs.iter_hprs(progressbar=True):
    if len(data['scan']) > 4_000:
        break

for hpr_idx, cycle, step, data in hprs.iter_hprs(progressbar=True):
    pass

columns = ("scan","tof","intensity")

%%timeit
empty_data = {col: np.array([], dtype=np.uint32) for col in columns}


X = {col: np.array([], dtype=np.uint32) for col in columns}
%%timeit
Y = {col: X[col] for col in columns}



counter = collections.Counter()
for hpr_idx, cycle, step, data in hprs.iter_hprs(progressbar=True):
    counter[len(data['scan'])] += 1
from clusterMS.stats import discrete_histogram
discrete_histogram(counter)

for hpr_idx, cycle, agg_data in hprs.iter_cycle_aggregate_hprs(progressbar=True):
    pass

%%snakeviz
for hpr_idx, cycle, diagonal, data in itertools.islice(hpr_transform(progressbar=True), 1_000):
    pass


# for (hpr_idx_old, cycle_old, agg_data_old), (hpr_idx_new, cycle_new, agg_data_new) in zip(
#     hprs.iter_cycle_aggregate_hprs(progressbar=True),
#     iter_cycle_aggregate_hprs(self)
# ):
#     assert hpr_idx_old == hpr_idx_new, f"hpr_idx {hpr_idx_new}"
#     assert cycle_old == cycle_new, f"hpr_idx {cycle_old}"
#     assert_data_the_same(agg_data_old, agg_data_new)


# for _ in self.iter_hprs(hpr_indices, progressbar):
#     pass

for hpr_idx, cycle, diagonal, data in itertools.islice(hpr_transform(progressbar=True), None):
    pass
# %%snakeviz
for hpr_idx, cycle, diagonal, data in itertools.islice(hpr_transform(progressbar=True), 1_000):
    # print(hpr_idx, cycle, diagonal)
    pass


for hpr_idx, cycle, diagonal, data in hpr_transform(progressbar=True):
    pass

from kilograms import scatterplot_matrix
scatterplot_matrix(pd.DataFrame(data))



it = hpr_transform()
hpr_idx, cycle, diagonal, data = next(it)
hpr_idx, cycle, diagonal
pd.DataFrame(data)
is_sorted(np.lexsort([data["tof"], data["scan"]]))


for i, group in enumerate(groups_tags):
    diagonal_to_datasets[group].append()




for hpr_idx, cycle, step, data in self.iter_hprs(hpr_indices, progressbar):



# who the fuck starts counting from 1 ???



x = groups == 3


# question: scan 163: should it be in or not? 
from savetimspy.numba_helper import binary_search

self = hprs
meta = next(metas)
frame_data["scan"][min_idx:max_idx]

# A quick control question for quality check: yes, we know that horizontally, i.e. per scan, there should be exactly 3 or 2 or 1 groups (depending on the position). Is there any kind of  obvious count vertically? Like, here is the distribution for HPR=100 of the 
collections.Counter(_M-_m+1 for _m, _M in self.hpr_step_to_scan_min_max.values())




