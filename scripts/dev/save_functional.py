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
FramesTable = pd.DataFrame(rawdata.frames)
# df = pd.DataFrame(
#     rawdata.query(frames=[13,14,15,16]),
#     columns=("frame","scan","tof","intensity"),
# )
# df = pd.concat([df]*1, ignore_index=True)
# original_frame_to_frame = {13:1, 14:2, 15:3, 16:4}
# df.frame = df.frame.map(original_frame_to_frame)
# df.frame = df.frame.astype(np.uint32)
# frame_to_original_frame = {1:13, 2:14, 3:15, 4:16}

verbose = True
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


(df.values == out.values).all()

wanted_contents_of_anlysis_tdf = pd.DataFrame(rawdata_source.frames).query(f"Id in {list(frame_to_original_frame.values())}")
cols_that_can_change = ("TimsId", "Id", "SummedIntensities", "MaxIntensity", "AccumulationTime")
cols = [c for c in wanted_contents_of_anlysis_tdf.columns if c not in cols_that_can_change]

before = wanted_contents_of_anlysis_tdf[cols].reset_index(drop=True)
after = pd.DataFrame(rawdata_target.frames)[cols]
before == after


def test_


get_random_df(10_000)