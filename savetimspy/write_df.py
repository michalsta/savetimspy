import numpy as np
import pandas as pd
import pathlib
import shutil
import sqlite3
import tqdm

from opentimspy.sql import table2dict
from typing import Dict, Union

from savetimspy.byte_ops import dump_one_ready_frame_df_to_tdf
from savetimspy.pandas_ops import (
    deduplicate,
    iter_group_based_views_of_data,
)


def write_df(
    df: pd.DataFrame,
    frame_to_original_frame: Dict[int,int],
    source: pathlib.Path,
    target: pathlib.Path,
    FramesTable: Union[pd.DataFrame, None]=None,
    run_deduplication_and_sorting: bool=True,
    verbose: bool=False,
) -> pathlib.Path:
    """
    Write a pandas data frame 'df' to the tdf file. 

    'df' should contain columns 'frame','scan','tof' and 'intensity'. 
    The procedure copies the entries of the old analysis.tdf and replaces the Frame table with a subset of the old entries precised by the user in `frame_to_original_frame`.

    Arguments:
        df (pd.DataFrame): Data to dump.
        frame_to_original_frame (dict): A mapping between the entries of df.frame and the original frame numbers whose meta information should be used in the new 'analysis.tdf'.
        source (pathlib.Path): The source folder: must contain 'analysis.tdf'.
        target (pathlib.Path): The target folder: must not exist before calling the function.
        FramesTable (pd.DataFrame|None): The original Frames table. If 'None' will be extracted from 'source/analysis.tdf'.
        run_deduplication_and_sorting (bool): Should we sum the intensities of events in the table with the same values of 'frame', 'scan', and 'tof'.
        verbose (bool): Print more about what is currently done to STDOUT.

    Return:
        pathlib.Path: Target .d folder with 'analysis.tdf' and 'analysis.tdf_bin'.
    """

    assert set(frame_to_original_frame) == set(df.frame.unique()), f"The mapping between frame to original frames does not contain the same entries as df.frame! It is {set(frame_to_original_frame)}, while df contains {set(df.frame.unique())}"

    target.mkdir()
    shutil.copyfile(source/'analysis.tdf', target/'analysis.tdf')

    if verbose:
        print("Creating entries of the new 'analysis.tdf'.")
    if FramesTable is None:
        FramesTable = pd.DataFrame(table2dict(source/'analysis.tdf', 'Frames'))
    FramesTable.set_index("Id", inplace=True)
    final_frames_df = FramesTable.loc[frame_to_original_frame.keys()].copy()
    frame_to_NumScans = final_frames_df.NumScans
    final_frames_df = final_frames_df.reset_index()
    final_frames_df.Id = range(1, len(final_frames_df)+1)
    with sqlite3.connect(target/"analysis.tdf") as tdf:
        tdf.execute("DELETE FROM Frames;")
        final_frames_df.to_sql(
            name="Frames",
            con=tdf,
            if_exists="append",
            index=False)# existing types and schemas remained

    if verbose:
        print("Creating entries of the new 'analysis.tdf_bin'.")
    for column_name in df:# this can make a copy
        if df[column_name].dtype != np.uint32:
            df[column_name] = df[column_name].astype(np.uint32)
    if verbose:
        print(f"Column Types are now:\n{df.dtypes}")

    if run_deduplication_and_sorting:
        if verbose:
            print("Deduplicating.")
        df = deduplicate(
            df=df,
            key_columns="frame scan tof".split(),
            sort=True)# sums intensities of multiple events 

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
            dump_one_ready_frame_df_to_tdf(
                frame_df=frame_df,
                open_file_handler=tdf_bin,
                total_scans=frame_to_NumScans[frame])
    if verbose:
        print(f"Finished with {target}")

    return target