%load_ext autoreload
%autoreload 2
import pandas as pd
pd.set_option('display.max_columns', None)
import pathlib

import opentimspy
from opentimspy.sql import table2dict

from savetimspy.write_df import write_df

source = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
target = pathlib.Path("tests/output.d")
rawdata = opentimspy.OpenTIMS(source)
FramesTable = pd.DataFrame(rawdata.frames)
compression_level = 1
df = pd.DataFrame(
    rawdata.query(frames=[13,14,15]),
    columns=("frame","scan","tof","intensity"),
)
df = pd.concat([df]*100, ignore_index=True)
original_frame_to_frame = {13:1, 14:2, 15:3}
df.frame = df.frame.map(original_frame_to_frame)
verbose = True
frame_to_original_frame = {1:13, 2:14, 3:15}

write_df(
    df=df,
    frame_to_original_frame=frame_to_original_frame,
    source=source,
    target=target,
    compression_level=compression_level,
    FramesTable=FramesTable,
    verbose=verbose
)
