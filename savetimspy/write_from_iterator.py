import pathlib

from savetimspy import SaveTIMS
from opentimspy import OpenTIMS
from collections import namedtuple

from savetimspy.common_assertions import (
    assert_minimal_input_for_clusterings_exist,
)


FrameDataset = namedtuple(
	"FrameDataset",
	"scan_number total_scans source_frame scans tofs intensities"
)


def write_from_iterator(
    frame_datasets: Iterable[FrameDataset],
	source: pathlib.Path,
    target: pathlib.Path,
    compression_level: int=1,
    set_MsMsType_to_0: bool=False,
    verbose: bool=False,
) -> pathlib.Path:
	try:
		if verbose:
			from tqdm import tqdm
			frame_datasets = tqdm(frame_datasets)

		with OpenTIMS(source) as ot,\
		     SaveTIMS(ot, target, compression_level) as saviour:
			for frame_dataset in frame_datasets:
				saviour.save_frame_tofs(
					set_MsMsType_to_0=set_MsMsType_to_0,
					**frame_dataset._as_dict(),
				)

	    if verbose:
		    print(f"Finished with: {target}")

	except FileExistsError:
        assert_minimal_input_for_clusterings_exist(target)
        if verbose:
            print(f"Results already there ({target}): not repeating.")        


