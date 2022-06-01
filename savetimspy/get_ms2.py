import numpy as np
import pandas as pd
import pathlib
import typing

from dia_common import DiaRun, parameters
from tqdm import tqdm

from savetimspy.get_frames import write_frames
from savetimspy.write_from_iterator import (
    FrameDataset,
    write_from_iterator
)

def write_ms2(
    source: pathlib.Path,
    target: pathlib.Path,
    compression_level: int=1,
    make_all_frames_seem_unfragmented: bool=True,
    verbose: bool=False,
) -> pathlib.Path:
    """Write all MS2 frames from source .d folder into target .d folder.
    
    Note: in comparison to collate, here we dump frame information from MS2 frames.

    Arguments:
        source (pathlib.Path): Path of the source .d folder.
        target (pathlib.Path): Path of the target .d folder: must not exist, parents must exist.
        compression_level (int): Final compression level.
        make_all_frames_seem_unfragmented (bool): Change the type of all reported frames in the .tdf so as to make them appear to be unfragmented, i.e. by setting MsMsType to 0.
        verbose (bool): Show progress bar.

    Returns:
        pathlib.Path: Path to the target.
    """
    diarun = DiaRun(
        fromwhat=source,
        preload_data=False,
        columns=parameters.indices_columns)

    write_frames(
        source=source,
        target=target,
        frame_indices=diarun.DiaFrameMsMsInfo.Frame.values,# all frames
        make_all_frames_seem_unfragmented=make_all_frames_seem_unfragmented,
        run_deduplication=False,
        verbose=verbose,
    )

    return target



def combined_ms2_frames_generator(
    source: pathlib.Path,
    verbose: bool=False,
) -> typing.Iterator[FrameDataset]:
    dia_run = DiaRun(source)
    NumScans = dia_run.Frames.NumScans.values
    cycles = range(dia_run.no_cycles)
    if verbose:
        cycles = tqdm(cycles)
    for cycle in cycles:
        ms2_frames_in_cycle = tuple(
            dia_run.cycle_step_to_ms2_frame(cycle, step)
            for step in range(dia_run.min_step, dia_run.max_step+1) 
        )
        df = pd.DataFrame(dia_run.opentims.query(
            ms2_frames_in_cycle,
            columns=["scan","tof","intensity"]
        ))
        df = df.groupby(
            ["scan","tof"],
            sort=True,
            as_index=False
        ).intensity.sum()
        ms1_frame_id = dia_run.cycle_to_ms1_frame(cycle)
        yield FrameDataset(
            scans=df.scan.values,
            tofs=df.tof.values,
            intensities=df.intensity.values,
            total_scans=NumScans[ms1_frame_id-1],
            src_frame=ms1_frame_id,
        )




if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='Extract a set of frames from a TDF dataset.')
    parser.add_argument("source", metavar="<source.d>", help="source path", type=pathlib.Path)
    parser.add_argument("target", metavar="<destination.d>", help="destination path", type=pathlib.Path)
    parser.add_argument("--force", "-f", help="force overwriting of the target path if it exists", action='store_true')
    parser.add_argument("--leave_original_meta", help="Leave all of the original contents of the analysis.tdf", action='store_true')
    parser.add_argument("--silent", "-s", help="do not display progressbar", action='store_true')
    parser.add_argument("--compression_level", help="Compression level used.", default=1, type=int)
    args = parser.parse_args()

    if args.force:
        shutil.rmtree(args.target, ignore_errors=True)

    write_ms2(
        source=args.source,
        target=args.target,
        compression_level=args.compression_level,
        make_all_frames_seem_unfragmented=not args.leave_original_meta,
        verbose=not args.silent,
    )
