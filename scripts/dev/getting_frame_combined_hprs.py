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
    max_mz=413,
)

hprs = HPRS(
    HPR_intervals=HPR_intervals,
    dia_run=DiaRun(source),
    verbose=True,
)


hprs.plot_scans_and_steps_used_by_hypothetical_precursor_range(hpr_idx=0)
it = iter(hprs)

it = hprs.full_iter()
hpr_byte = next(it)
print(hpr_byte.cycle, hpr_byte.step, hpr_byte.hpr_idx)
len(hpr_byte.intensities)


df = pd.DataFrame({
    "scan":         hpr_byte.scans,
    "tof":          hpr_byte.tofs,
    "intensity":    hpr_byte.intensities
})
df


