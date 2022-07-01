import numpy as np
import pandas as pd
import pathlib
import typing

from dia_common import DiaRun, parameters
from tqdm import tqdm

from savetimspy.get_frames import write_frames
from savetimspy.write_frame_datasets import (
    FrameSaveBundle,
    write_frame_datasets
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
        frame_indices=diarun.DiaFrameMsMsInfo.Frame.to_numpy(),# all frames
        make_all_frames_seem_unfragmented=make_all_frames_seem_unfragmented,
        run_deduplication=False,
        verbose=verbose,
    )

    return target



def combined_ms2_frames_generator(
    source: pathlib.Path,
    verbose: bool=False,
) -> typing.Iterator[FrameSaveBundle]:
    dia_run = DiaRun(source)
    NumScans = dia_run.Frames.NumScans.to_numpy()
    dedup = lambda df: df.groupby(
            ["scan","tof"],
            sort=True,
            as_index=False
        ).intensity.sum()
    meta_per_cycle = dia_run.DiaFrameMsMsInfo.groupby("cycle")
    if verbose:
        meta_per_cycle = tqdm(meta_per_cycle)
    for cycle, meta in meta_per_cycle:
        df = pd.DataFrame(dia_run.opentims.query(
            meta.Frame,
            columns=["scan","tof","intensity"]
        ))
        df = dedup(df)
        ms1_frame_id = dia_run.cycle_to_ms1_frame(cycle)
        yield FrameSaveBundle(
            total_scans=NumScans[ms1_frame_id-1],
            src_frame=ms1_frame_id,
            scans=df.scan.to_numpy(),
            tofs=df.tof.to_numpy(),
            intensities=df.intensity.to_numpy(),
        )


def write_ms2_combined_diagonals(
    source: pathlib.Path,
    target: pathlib.Path,
    compression_level: int=1,
    make_all_frames_seem_unfragmented: bool=True,
    verbose: bool=False,
) -> pathlib.Path:
    """Write all MS2 frames from source .d folder into target .d folder.
    
    Note: this likely works as collate:
        For each cycle, we extract the MS2 steps, combine them, deduplicate,
        and dump to a tdf.
        Each such dumped frame gets the meta information of the MS1 frame in that cycle.

    Arguments:
        source (pathlib.Path): Path of the source .d folder.
        target (pathlib.Path): Path of the target .d folder: must not exist, parents must exist.
        compression_level (int): Final compression level.
        make_all_frames_seem_unfragmented (bool): Change the type of all reported frames in the .tdf so as to make them appear to be unfragmented, i.e. by setting MsMsType to 0.
        verbose (bool): Show progress bar.

    Returns:
        pathlib.Path: Path to the target.
    """
    return write_frame_datasets(
        frame_datasets=combined_ms2_frames_generator(source, verbose=verbose),
        source=source,
        target=target,
        set_MsMsType_to_0=make_all_frames_seem_unfragmented,
        run_deduplication=False,
    )


def cli():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='Extract a set of MS2 frames from a TDF dataset.')
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


if __name__ == "__main__":
    cli()