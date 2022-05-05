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
import shutil


for _ in range(1000):
    source = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
    target = pathlib.Path("tests/output.d")
    try:
        shutil.rmtree(target)
    except FileNotFoundError:
        pass

    rawdata = opentimspy.OpenTIMS(source)

    df = get_random_df(
        size=10_000, 
        min_frame=1,
        max_frame=100,
        min_scan=400,
        max_scan=600,
        min_tof=0,
        max_tof=1000,
    )
    df = deduplicate(df)
    drawn_frames = np.sort(df.frame.unique())
    new2final={}
    for i, fr in enumerate(drawn_frames):
        new2final[fr] = i+1
    df.frame = df.frame.map(new2final).astype(np.uint32)

    for col in df:
        print(f"min({col})={df[col].min()}")
        print(f"max({col})={df[col].max()}")


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

    rawdata_target = opentimspy.OpenTIMS(target)

    out = pd.DataFrame(rawdata_target.query(
        frames=np.sort(df.frame.unique())),
        columns="frame scan tof intensity".split())

    if not (df == out).all().all():
        print("error!!!!")
        break

rawdata.query(frames=2, columns=['scan'])['scan'].max()
rawdata.frames["NumScans"][1]

rawdata.min_mz
rawdata.max_mz

