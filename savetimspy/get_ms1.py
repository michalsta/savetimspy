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
		verbose=verbose,
	)
	return output_ms1_d


