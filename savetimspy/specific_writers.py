import opentimspy
import pandas as pd
import pathlib
import numpy as np
import multiprocessing as mp

from opentimspy.sql import table2dict

from savetimspy.write_df import write_df
from savetimspy.pandas_ops import trivial_group_index_map
from savetimspy.numba_helper import get_groups_as_consecutive_ints


def extract_and_save_diagonal(
    window_group: int,
    window_frames,
    source,
    target,
) -> pathlib.Path:
    rawdata = opentimspy.OpenTIMS(source)
    df = pd.DataFrame(rawdata.query(
        frames=window_frames.Frame.values,
        columns="frame scan tof intensity".split()))
    frame_to_original_frame = trivial_group_index_map(df.frame.values)
    _ = get_groups_as_consecutive_ints(np.arange(10))#run llvm just in case...
    df.frame = get_groups_as_consecutive_ints(df.frame.values)
    target_path = write_df(
        df=df,
        frame_to_original_frame=frame_to_original_frame,
        source=source,
        target=target/f"MS2_MIDIA_STEP_{window_group-1}.d",
        _deduplicate=True,
        _sort=True,
        _verbose=False,
    ) 
    print(f"Finished with window group {window_group}.")
    return target_path


def assert_minimal_input_for_clusterings_exist(path: pathlib.Path):
    assert path.exists(), f"File {path} does not exist."
    tdf = path/"analysis.tdf"
    assert tdf.exists(), f"File {tdf} does not exist."
    tdf_bin = path/"analysis.tdf_bin"
    assert tdf_bin.exists(), f"File {tdf_bin} does not exist."



def write_diagonals(
    source: pathlib.Path,
    target: pathlib.Path,
    processesNo: int=10,
):
    DiaFrameMsMsInfo = pd.DataFrame(table2dict(source/"analysis.tdf", "DiaFrameMsMsInfo"))

    _input_stream = ((window_group, window_data, source, target)
     for window_group, window_data in DiaFrameMsMsInfo.groupby("WindowGroup"))
    try:
        target.mkdir()
        with mp.Pool(processesNo) as pool:
            target_paths = pool.starmap(extract_and_save_diagonal, _input_stream)
    except FileExistsError:
        steps = np.sort(DiaFrameMsMsInfo.WindowGroup.unique())-1
        target_paths = [target/f"MS2_MIDIA_STEP_{step}.d" for step in steps]
        for target_path in target_paths:
            assert_minimal_input_for_clusterings_exist(target_path)

    return target_paths