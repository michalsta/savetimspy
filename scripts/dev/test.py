%load_ext autoreload
%autoreload 2
import pandas as pd
pd.set_option('display.max_columns', None)
import pathlib


import opentimspy
import numpy as np
from opentimspy.sql import table2dict

from savetimspy.write_df import write_df
from savetimspy.pandas_ops import deduplicate

from savetimspy.writer import SaveTIMS
from savetimspy.random import get_random_df

source = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
target = pathlib.Path("tests/output.d")
rawdata = opentimspy.OpenTIMS(source)

df = deduplicate(get_random_df(10_000))
frame_to_original_frame = {}
old_frames = np.sort(df.frame.unique())
for new_frame, old_frame in zip(range(1, len(old_frames)+1), old_frames):
    frame_to_original_frame[new_frame] = old_frame

write_df(
    df=df,
    frame_to_original_frame=frame_to_original_frame,
    source=source,
    target=target,
    _deduplicate=True,
    _sort=True,
    _verbose=True,
) 

rawdata_source = opentimspy.OpenTIMS(source) 
rawdata_target = opentimspy.OpenTIMS(target)

out = pd.DataFrame(rawdata_target.query(
    frames=[1,2,3,4]),
    columns="frame scan tof intensity".split())

