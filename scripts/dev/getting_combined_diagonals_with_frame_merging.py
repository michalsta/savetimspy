%load_ext autoreload
%autoreload 2
from savetimspy.get_frames import write_frames
from savetimspy.get_ms1 import write_ms1
from savetimspy.get_ms2 import write_ms2
from savetimspy.get_diagonals import write_diagonals
from savetimspy.get_hprs import
import shutil
import pathlib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import sqlite3
import opentimspy


project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")
source = frag5P

target = project_folder/"tests"/"data"/"test_combined_diagonals.d"

from savetimspy import SaveTIMS
from opentimspy import OpenTIMS
from collections import namedtuple

from savetimspy.common_assertions import (
    assert_minimal_input_for_clusterings_exist,
)


# need some sort of data iterator: it should provide:
# scan_number, total_scans, source_frame, scans, tofs, intensities

ot = OpenTIMS(source)
FrameDataset = namedtuple(
	"FrameDataset",
	"scan_number total_scans source_frame scans tofs intensities"
)
compression_level = 1



def write_from_iterator(
	source: pathlib.Path,
    target: pathlib.Path,
    frame_datasets: Iterable[FrameDataset],
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


# shutil.rmtree(target)
# target.exists()


