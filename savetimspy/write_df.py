import numpy as np
import pandas as pd
import pathlib
import shutil
import sqlite3
import tqdm

from opentimspy.sql import table2dict
from typing import Dict, Union

from savetimspy.byte_ops import (
    get_data_as_bytes,
    compress_data,
    write_frame_bytearray_to_open_file,
)
from savetimspy.pandas_ops import (
    deduplicate,
    iter_group_based_views_of_data,
)


def write_df(
    df: pd.DataFrame,
    frame_to_original_frame: Dict[int,int],
    source: pathlib.Path,
    target: pathlib.Path,
    compression_level: int=1,
    FramesTable: Union[pd.DataFrame, None]=None,
    verbose: bool=True,
) -> None:

    assert set(frame_to_original_frame) == set(df.frame.unique()), f"The mapping between frame to original frames does not contain the same entries as df.frame! It is {set(frame_to_original_frame)}, while df contains {set(df.frame.unique())}"

    target.mkdir()
    shutil.copyfile(source/'analysis.tdf', target/'analysis.tdf')

    df = deduplicate(df, key_columns="frame scan tof".split(), sort=True)# merges double entries 
    if FramesTable is None:
        FramesTable = pd.DataFrame(table2dict(source/'analysis.tdf', 'Frames'))
    FramesTable.set_index("Id", inplace=True)

    final_frames_df = FramesTable.loc[frame_to_original_frame.keys()]
    frame_to_NumPeaks = final_frames_df.NumPeaks.copy()
    final_frames_df = final_frames_df.reset_index()
    final_frames_df.Id = range(1,len(final_frames_df)+1)
    if verbose:
        print(f"Saving {target/'analysis.tdf'}")
    with sqlite3.connect(target/"analysis.tdf") as tdf:
        # this does not touch existing types or their schema: a clear linux-linux
        tdf.execute("DELETE FROM Frames;")
        final_frames_df.to_sql(
            name="Frames",
            con=tdf,
            if_exists="append",
            index=False)

    if verbose:
        print(f"Dumping frame to {target/'analysis.tdf_bin'}")
    with open(target/"analysis.tdf_bin", "wb") as tdf_bin:
        frame_data_tuples = iter_group_based_views_of_data(
            df=df,
            grouping_column_name="frame",
            assert_sorted=False)
        if verbose:
            frame_data_tuples = tqdm.tqdm(
                frame_data_tuples,
                total=len(frame_to_original_frame))
        for frame, frame_df in frame_data_tuples:
            total_scans = int(frame_to_NumPeaks[frame])
            frame_bytes = get_data_as_bytes(
                scans = frame_df.scan.values,
                tofs = frame_df.tof.values,
                intensities = frame_df.intensity.values,
                total_scans = total_scans)
            compressed_data = compress_data(
                data=frame_bytes,
                compression_level=compression_level)
            write_frame_bytearray_to_open_file(
                file=tdf_bin,
                data=compressed_data,
                total_scans=total_scans)

    if verbose:
        print(f"Finished with {target}")