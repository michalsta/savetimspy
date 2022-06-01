%load_ext autoreload
%autoreload 2
from savetimspy.get_frames import write_frames
from savetimspy.get_ms1 import write_ms1
from savetimspy.get_ms2 import write_ms2
from savetimspy.get_diagonals import write_diagonals
from savetimspy.get_hprs import
import shutil
import pathlib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import sqlite3
import opentimspy

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")

target = project_folder/"tests"/"data"/"test_writing_frames.d"
res = write_frames(
	source=unfrag5P,
	target=target,
	frame_indices=[2,3,4,10,22,23],
	make_all_frames_seem_unfragmented=False,
	verbose=True,
)
shutil.rmtree(target)
target.exists()


