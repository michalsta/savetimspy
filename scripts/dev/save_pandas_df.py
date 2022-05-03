# %load_ext autoreload
# %autoreload 2
# from savetimspy.merging_groups import *
import pathlib
import opentimspy
import shutil
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import tqdm
import zstd
import sqlite3

from savetimspy.byte_ops import (
    get_data_as_bytes,
    compress_data,
    write_frame_bytearray_to_open_file,
)
from savetimspy.writer import SaveTIMS, SaveTIMS2
from savetimspy.numba_helper import deduplicate as deduplicate_numba
from savetimspy.pandas_ops import (
    deduplicate,
    iter_group_based_views_of_data,
)
from opentimspy.sql import table2dict

source_tdf_folder = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
destination_folder = pathlib.Path("tests/output.d")
try:
    shutil.rmtree(destination_folder)
except FileNotFoundError:
    pass
verbose = True

rawdata = opentimspy.OpenTIMS(source_tdf_folder)


# saviour = SaveTIMS2(rawdata, destination_folder)

df = pd.DataFrame(
    rawdata.query(frames=[13,14,15]),
    columns=("frame","scan","tof","intensity"),
)
df = pd.concat([df]*100, ignore_index=True)
df = deduplicate(df)

# input
frame_indices_map = {13:1, 14:1, 15:2}
frame_indices_df = pd.DataFrame(
    frame_indices_map.items(),
    columns=("original_frame","new_frame")
)
frames_df = pd.DataFrame(rawdata.frames).set_index("Id")
final_frames_df = frames_df.loc[df.frame.unique()]
final_frames_df = final_frames_df.reset_index()
final_frames_df.Id = range(1,len(final_frames_df)+1)
final_frames_df.to_sql(
    name="Frames",
    con=saviour.sqlcon,
    if_exists="append",
    # schema='online',
    index=False,
)
frame_data_tuples = iter_group_based_views_of_data(
    df=df,
    grouping_column_name="frame",
    assert_sorted=False,
)
if verbose:
    frame_data_tuples = tqdm.tqdm(frame_data_tuples, total=len(frame_indices_map))

for frame, frame_df in frame_data_tuples:
    saviour.save_frame(
        scans = frame_df.scan.values,
        tofs = frame_df.tof.values,
        intensities = frame_df.intensity.values,
        total_scans = int(frames_df.loc[13].NumScans),
    )
saviour.close()





# qmarks = "?,"*len(final_frames_df.columns)
# qmarks = qmarks[:-1]
# rows = list(final_frames_df.itertuples(index=False))
# sql = "INSERT INTO `Frames` VALUES (" + qmarks + ")"
# with saviour.sqlcon as con:
#     gowno = con.executemany(
#         sql,
#         rows,
#     )
