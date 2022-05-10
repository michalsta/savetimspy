from __future__ import annotations
from opentimspy import OpenTIMS

import numpy as np
import numpy.typing as npt
import sqlite3
import typing
import pandas as pd
import pathlib

from savetimspy import SaveTIMS



def write_frames(
    source: pathlib.Path,
    target: pathlib.Path,
    frame_indices: typing.Union[npt.NDArray[np.uint32], int, typing.Iterable[int]],
    compression_level: int=1,
    make_all_frames_seem_unfragmented: bool=False,
    verbose: bool=False,
) -> pathlib.Path:
    """Write a selection of frames from source .d folder into target .d folder.
    
    Arguments:
        source (pathlib.Path): Path of the source .d folder.
        target (pathlib.Path): Path of the target .d folder: must not exist, parents must exist.
        frame_indices (int, Iterable[int]): Frames to extract.
        compression_level (int): Final compression level.
        make_all_frames_seem_unfragmented (bool): Change the type of all reported frames in the .tdf so as to make them appear to be unfragmented, i.e. by setting MsMsType to 0.
        verbose (bool): Show progress bar.

    Returns:
        pathlib.Path: Path to the target.
    """
    if not verbose:
        progressbar = lambda x: x
    else:
        from tqdm import tqdm as progressbar

    frame_indices = np.r_[frame_indices]

    with OpenTIMS(source) as ot,\
        SaveTIMS(ot, target, compression_level) as s,\
        sqlite3.connect(source/'analysis.tdf') as db:
        Id_to_NumScans = dict(zip(ot.frames["Id"], ot.frames["NumScans"]))
        for frame in progressbar(frame_indices):
            D = ot.query(frame, columns="scan tof intensity".split())
            n_scans = int(Id_to_NumScans[frame])
            # list(db.execute("SELECT NumScans FROM Frames WHERE Id == ?", (frame,)))[0][0]# this was empty!
            s.save_frame_tofs(
                scans=D['scan'],
                tofs=D['tof'],
                intensities=D['intensity'],
                total_scans=n_scans,
            )

    if make_all_frames_seem_unfragmented:
        with sqlite3.connect(target/'analysis.tdf') as dst_db:
            dst_db.execute("UPDATE Frames set MsMsType=0;")

    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract a set of frames from a TDF dataset.')
    parser.add_argument("in_p", metavar="<source.d>", help="source path")
    parser.add_argument("out_p", metavar="<destination.d>", help="destination path")
    parser.add_argument("frames", help="comma-separated list of frames to extract. May contain ranges. Example: 314,317-320,350")
    parser.add_argument("--force", "-f", help="force overwriting of the target path if it exists", action='store_true')
    parser.add_argument("--ms1", help="mark all frames as ms1", action='store_true')
    parser.add_argument("--silent", "-s", help="do not display progressbar", action='store_true')
    args = parser.parse_args()

    src = pathlib.Path(args.in_p)
    dst = pathlib.Path(args.out_p)

    frames = set()
    for frame_desc in args.frames.split(','):
        if '-' in frame_desc:
            start, end = frame_desc.split('-')
            frames.update(range(int(start), int(end)+1))
        else:
            frames.add(int(frame_desc))
    frames = sorted(frames)

    if args.force:
        shutil.rmtree(dst, ignore_errors=True)
    
    assert not dst.exists(), f"Folder {dst} already exists: remove it or add '--force' to your command, Luke."


    _ = write_frames(
        source=src,
        target=dst,
        frame_indices=frames,
        verbose=not args.silent,
        make_all_frames_seem_unfragmented=args.ms1,
    )

    if not args.silent:
        print(f"Finished with: {dst}")



