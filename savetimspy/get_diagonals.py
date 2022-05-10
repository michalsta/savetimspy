import opentimspy
import pandas as pd
import pathlib
import numpy as np
import multiprocessing as mp

from typing import List
from opentimspy.sql import table2dict
from dia_common import DiaRun, parameters

from savetimspy.get_frames import write_frames


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
    compression_level: int=1,
    make_all_frames_seem_unfragmented: bool=True,
    verbose: bool=False,
) -> List[pathlib.Path]:
    """Write each of the MIDIA diagonal into a respective subfolder of the target folder.
    
    The already existing results are reused (disk-based-caching).
    
    Note: in comparison to collate, here we dump frame information from MS2 frames.

    Arguments:
        source (pathlib.Path): Path of the source .d folder.
        target (pathlib.Path): Path of the target .d folder: must not exist, parents must exist.
        compression_level (int): Final compression level.
        make_all_frames_seem_unfragmented (bool): Change the type of all reported frames in the .tdf so as to make them appear to be unfragmented, i.e. by setting MsMsType to 0.
        verbose (bool): Add verbosity.

    Returns:
        pathlib.Path: Path to the target.
    """

    # Comment: yes, this does not use DiaRun, simply to avoid one additional opening of the rawdata.
    diarun = DiaRun(fromwhat=source,
                    preload_data=False,
                    columns=parameters.indices_columns)

    _input_stream = (
        (
            source,
            target/f"MS2_MIDIA_STEP_{step}.d",#target
            frames.values,# frame indices
            compression_level,
            make_all_frames_seem_unfragmented,
            False,#verbose: does not play well with multiprocessing ;)
        )
        for step, frames in diarun.DiaFrameMsMsInfo.groupby("step").Frame
    )
    processesNo = min(
        processesNo,
        len(diarun.DiaFrameMsMsInfo.step.unique())
    )
    try:
        target.mkdir()
        if verbose:
            print(f"Running {processesNo} processes to make tdfs.")
        with mp.Pool(processesNo) as pool:
            target_paths = pool.starmap(write_frames, _input_stream)
    except FileExistsError:
        target_paths = [target/f"MS2_MIDIA_STEP_{step}.d" for step in np.sort(diarun.DiaFrameMsMsInfo.step.unique())]
        for target_path in target_paths:
            assert_minimal_input_for_clusterings_exist(target_path)
        if verbose:
            print(f"Results were already there: not repeating.")

    return target_paths



if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='Extract a set of frames from a TDF dataset.')
    parser.add_argument("source", metavar="<source.d>", help="source path", type=pathlib.Path)
    parser.add_argument("target", metavar="<destination.d>", help="destination path", type=pathlib.Path)
    parser.add_argument("--leave_original_meta", help="Leave all of the original contents of the analysis.tdf", action='store_true')
    parser.add_argument("--verbose", 
        action='store_true',
        help='Print more info to stdout.')
    parser.add_argument("--compression_level", help="Compression level used.", default=1, type=int)
    parser.add_argument("--processes", help="Upper boundry on the number of processes in multiprocessing to use. The actual number will be this or the number of diagonals.", default=10, type=int)
    args = parser.parse_args()

    target_paths = write_diagonals(
        source=args.source,
        target=args.target,
        processesNo=args.processes,
        compression_level=args.compression_level,
        make_all_frames_seem_unfragmented=not args.leave_original_meta,
        verbose=args.verbose
    )
    if args.verbose:
        print(f"Outcome .d folders:\n"+"\n".join(str(tp) for tp in target_paths))


# def extract_and_save_diagonal(
#     window_group: int,
#     window_frames: np.uint32,
#     source: pathlib.Path,
#     target: pathlib.Path,
#     compression_level: int=1,
#     make_all_frames_seem_unfragmented: bool=True,
# ) -> pathlib.Path:
#     rawdata = opentimspy.OpenTIMS(source)
#     df = pd.DataFrame(rawdata.query(
#         frames=window_frames,
#         columns="frame scan tof intensity".split()))

#     frame_to_original_frame = trivial_group_index_map(df.frame.values)
#     _ = get_groups_as_consecutive_ints(np.arange(10))#run llvm just in case...

#     df.frame = get_groups_as_consecutive_ints(df.frame.values)
#     target_path = write_df(
#         df=df,
#         frame_to_original_frame=frame_to_original_frame,
#         source=source,
#         target=target/f"MS2_MIDIA_STEP_{window_group-1}.d",
#         _deduplicate=True,
#         _sort=True,
#         _verbose=False,
#     ) 
#     print(f"Finished with step {window_group}.")
#     return target_path
