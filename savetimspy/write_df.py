import numpy as np
import pandas as pd
import pathlib
import shutil
import sqlite3
import tqdm
import opentimspy

from opentimspy.sql import table2dict
from typing import Dict, Union

from savetimspy.writer import SaveTIMS
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
    make_all_frames_seem_unfragmented: bool=True,
    _FramesTable: Union[pd.DataFrame, None]=None,
    _deduplicate: bool=True,
    _sort: bool=True,
    _verbose: bool=False,
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
        compression_level (int): Final compression level.
        make_all_frames_seem_unfragmented (bool): Change the type of all reported frames in the .tdf so as to make them appear to be unfragmented, i.e. by setting MsMsType to 0.
        _FramesTable (pd.DataFrame|None): The original Frames table. If 'None' will be extracted from 'source/analysis.tdf'.
        _deduplicate: (bool): Should we sum the intensities of events in the table with the same values of 'frame', 'scan', and 'tof'?
        _sort (bool): Should we sort df?
        _verbose (bool): Print more about what is currently done to STDOUT.

    Return:
        pathlib.Path: Target .d folder with 'analysis.tdf' and 'analysis.tdf_bin'.
    """

    assert set(frame_to_original_frame) == set(df.frame.unique()), f"The mapping between frame to original frames does not contain the same entries as df.frame!"

    input_rawdata = opentimspy.OpenTIMS(source)

    if _FramesTable is None:
        _FramesTable = pd.DataFrame(input_rawdata.table2dict('Frames'))
    if _FramesTable.index.name != "Id":
        _FramesTable.set_index("Id", inplace=True)
    frame_to_NumScans = _FramesTable.NumScans[frame_to_original_frame.keys()]

    for column_name in df:# this can make a copy
        if df[column_name].dtype != np.uint32:
            df[column_name] = df[column_name].astype(np.uint32)

    if _deduplicate:
        if _verbose:
            print("Deduplicating and sorting.")
        df = deduplicate(
            df=df,
            key_columns="frame scan tof".split(),
            sort=_sort)# sums intensities of multiple events 
    else:
        if _sort:
            df.sort_values(by="frame scan tof".split(), inplace=True)

    frame_data_tuples = iter_group_based_views_of_data(
        df=df,
        grouping_column_name="frame",
        assert_sorted=False)
    if _verbose:
        frame_data_tuples = tqdm.tqdm(
            frame_data_tuples,
            total=len(frame_to_original_frame))
    with SaveTIMS(
        opentims_obj=input_rawdata,
        path=target,
        compression_level=compression_level,
    ) as saviour:
        for frame, frame_df in frame_data_tuples:
            if frame_df.scan.max() > frame_to_NumScans[frame]:
                raise Exception(f"Submitted scan ({frame_df.scan.max()}) was beyond value in the original '.tdf' as total_scans ({frame_to_NumScans[frame]}). Stoping dumping. Results are kept on disk under {target}.")
            saviour.save_frame_tofs(
                scans=frame_df.scan.values,
                tofs=frame_df.tof.values,
                intensities=frame_df.intensity.values,
                total_scans=int(frame_to_NumScans[frame]),
                copy_sql=int(frame_to_original_frame[frame]),
                run_deduplication=False,
                set_MsMsType_to_0=make_all_frames_seem_unfragmented,
            )

    if _verbose:
        print(f"Finished with {target}")

    return target

# Making these tests is shit.
# def test_writer_df():
#     import pkg_resources
#     from savetimspy.random import get_random_df
#     src = pathlib.Path(pkg_resources.resource_filename("savetimspy", "data/test.d"))
#     target = pathlib.Path("/tmp/test_writer_df.d")
#     df = get_random_df(10_000)
    
#     frame_to_original_frame = {}
#     old_frames = np.sort(df.frame.unique())
#     for new_frame, old_frame in zip(range(1, len(old_frames)+1), old_frames):
#         frame_to_original_frame[new_frame] = old_frame

#     write_df(
#         df=df,
#         frame_to_original_frame=frame_to_original_frame,
#         source=src,
#         target=target,
#         _deduplicate=True,
#         _sort=True,
#         _verbose=False,
#     )
