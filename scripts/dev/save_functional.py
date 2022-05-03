import pandas as pd
pd.set_option('display.max_columns', None)
import pathlib
import shutil
import sqlite3

from savetimspy.byte_ops import (
    get_data_as_bytes,
    compress_data,
    write_frame_bytearray_to_open_file,
)


compression_level = 1
source_folder = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
target_folder = pathlib.Path("tests/output.d")

target_folder.mkdir()
shutil.copyfile(
    source_folder/'analysis.tdf',
    target_folder/'analysis.tdf',
)

with open(target_folder/"analysis.tdf_bin", "wb") as tdf_bin:
    # frame dumping
    frame_data_tuples = iter_group_based_views_of_data(
        df=df,
        grouping_column_name="frame",
        assert_sorted=False,
    )
    if verbose:
        frame_data_tuples = tqdm.tqdm(frame_data_tuples, total=len(frame_indices_map))
    for frame, frame_df in frame_data_tuples:
        total_scans = ?
        frame_bytes = get_data_as_bytes(
            scans = frame_df.scan.values,
            tofs = frame_df.tof.values,
            intensities = frame_df.intensity.values,
            total_scans = total_scans,
        )
        compressed_data = compress_data(
            data=frame_bytes,
            compression_level=compression_level
        )
        write_frame_bytearray_to_open_file(
            file=tdf_bin,
            total_scans=total_scans,
        )

# sqlite meta info preparation
with sqlite3.connect(target_folder/"analysis.tdf") as tdf:
    pass