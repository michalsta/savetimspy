import pathlib
from dia_common import DiaRun, parameters

from savetimspy.get_frames import write_frames


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
        verbose=verbose,
        make_all_frames_seem_unfragmented=make_all_frames_seem_unfragmented)

    return target


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
