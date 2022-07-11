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


for _ in hprs.iter_hprs(progressbar=True, hpr_indices=[100]):
    pass
for _ in hprs.iter_hpr_transform(progressbar=True, hpr_indices=[100]):
    pass

# initially slow because of the matrix computations
for _ in hprs.iter_hpr_transform(progressbar=True):
    pass


hprs.plot_scans_and_steps_used_by_hypothetical_precursor_range(100)

from kilograms import scatterplot_matrix
scatterplot_matrix(pd.DataFrame(data))


is_sorted(np.lexsort([data["tof"], data["scan"]]))


for i, group in enumerate(groups_tags):
    diagonal_to_datasets[group].append()





