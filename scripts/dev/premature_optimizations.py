%load_ext autoreload
%autoreload 2
%load_ext snakeviz
import shutil
import pathlib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import sqlite3

from dia_common import DiaRun
from tqdm import tqdm
from savetimspy.writer import SaveTIMS, update_frames_table

import numpy as np

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
unfrag5P = _get_d("*3517.d")

cols = ("scan", "tof", "intensity")
diarun = DiaRun(unfrag5P, preload_data=False, columns=("frame",)+cols)
total_scans = int(diarun.Frames.NumScans.max())
target = "/tmp/test.d"
src_frames = diarun.Frames.query("MsMsType == 0").Id.to_numpy()


with SaveTIMS(diarun.opentims, target) as saviour:
    for frame_id in range(1, 100):
        data = diarun.opentims.query(frame_id, columns=cols)
        saviour.save_frame_tofs(
            scans=data['scan'],
            tofs=data['tof'],
            intensities=data['intensity'],
            total_scans=total_scans,
            src_frame=src_frames[frame_id-1],
            run_deduplication=True,
            MsMsType=0
        )

