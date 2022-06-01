%load_ext autoreload
%autoreload 2
from savetimspy.get_frames import write_frames
from savetimspy.get_ms1 import write_ms1
from savetimspy.get_ms2 import write_ms2, combined_ms2_frames_generator
from savetimspy.get_diagonals import write_diagonals
from savetimspy.get_hprs import write_hprs
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

from savetimspy.write_from_iterator import (
    FrameDataset,
    write_from_iterator
)

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")
source = frag5P
source.exists()

target = project_folder/"tests"/source.name

write_frame_datasets(
    frame_datasets=combined_ms2_frames_generator(source, verbose=True),
    source=source,
    target=target,
    set_MsMsType_to_0=True,
    run_deduplication=False,
)
features = Run4DFFv4_12_1(target)
clusters = read_4DFF_to_df_physical_dims_only(features)

_get_clusters = lambda p: read_4DFF_to_df_physical_dims_only(p/f"{p.name}.features")
ms1_vanilla = _get_clusters(source)
# shutil.rmtree(target)
# target.exists()


