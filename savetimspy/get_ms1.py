from __future__ import annotations
import pathlib

from dia_common import DiaRun
from savetimspy.get_frames import write_frames


def write_ms1(
	source: pathlib.Path|str,
	target: pathlib.Path|str|None = None,
	compression_level: int = 1,
	verbose: bool = False,
):
	"""Write a tdf with all MS1 frames.

	This is mostly done for checking if 4DFF works as predicted.


	Arguments:
        source (pathlib.Path): Path of the source .d folder.
        target (pathlib.Path): Path of the target .d folder: must not exist, parents must exist.
        compression_level (int): Final compression level.
        make_all_frames_seem_unfragmented (bool): Change the type of all reported frames in the .tdf so as to make them appear to be unfragmented, i.e. by setting MsMsType to 0.
        verbose (bool): Show progress bar.

	Returns:
        pathlib.Path: Path to the target.
	"""
	source = pathlib.Path(source)
	if target is None:
		target = source/"ms1.d"
	dia_run = DiaRun(source)
	output_ms1_d = write_frames(
		source=source,
		target=target,
		frame_indices=dia_run.Frames.query("MsMsType == 0").Id.values,
		compression_level=compression_level,
		make_all_frames_seem_unfragmented=True,
		run_deduplication=False,
		verbose=verbose,
	)
	return output_ms1_d


def cli():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='Extract a set of MS1 frames from a TDF dataset.')
    parser.add_argument("source", metavar="<source.d>", help="source path", type=pathlib.Path)
    parser.add_argument("target", metavar="<destination.d>", help="destination path", type=pathlib.Path)
    parser.add_argument("--force", "-f", help="force overwriting of the target path if it exists", action='store_true')
    parser.add_argument("--compression_level", help="Compression level used.", default=1, type=int)
    parser.add_argument("--silent", "-s", help="do not display progressbar", action='store_true')
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