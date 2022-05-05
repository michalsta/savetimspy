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

source = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
target = pathlib.Path("tests/output.d")
rawdata = opentimspy.OpenTIMS(source)
FramesTable = pd.DataFrame(rawdata.frames)
df = pd.DataFrame(
    rawdata.query(frames=[13,14,15]),
    columns=("frame","scan","tof","intensity"),
)
df = pd.concat([df]*100, ignore_index=True)
original_frame_to_frame = {13:1, 14:2, 15:3}
df.frame = df.frame.map(original_frame_to_frame)
df.frame = df.frame.astype(np.uint32)
df.dtypes





verbose = True
frame_to_original_frame = {1:13, 2:14, 3:15}

write_df(
    df=df,
    frame_to_original_frame=frame_to_original_frame,
    source=source,
    target=target,
    FramesTable=FramesTable,
    verbose=verbose,
)

deduplicate(df)

rawdata_final = opentimspy.OpenTIMS(target)
rawdata_final.query(frames=[1,2,3])


from savetimspy.pandas_ops import (
    deduplicate,
    iter_group_based_views_of_data,
)

df = deduplicate(df)
df.dtypes

x = open("/tmp/test","wb")
type(x)
