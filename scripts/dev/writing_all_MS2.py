%load_ext autoreload
%autoreload 2

"""This all can be optimized later on by specific uses of savetimspy."""
# from savetimspy.specific_writers import write_diagonals
import pathlib
import opentimspy
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import multiprocessing as mp

from dia_common import DiaRun, parameters
from savetimspy.get_frames import write_frames
from savetimspy.get_ms2 import write_ms2

source = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
target = pathlib.Path("tests/midia_fragments.d")

write_ms2(
    source=source,
    target=target,
    compression_level=1,
    make_all_frames_seem_unfragmented=True,
    verbose=True,
)

diarun = DiaRun(source, preload_data=False, columns=parameters.indices_columns)

opentimspy = OpenTIMS(source)
write_frames(
    source=source,
    target=target,
    frame_indices=diarun.DiaFrameMsMsInfo.Frame.values[:100],
    verbose=True,
    make_all_frames_seem_unfragmented=True,
)

# db = sqlite3.connect(source/'analysis.tdf')
# list(db.execute("SELECT NumScans FROM Frames WHERE Id == ?", (frame,)))
# db.close()
from dia_common import DiaRun, parameters
import pathlib
import opentimspy
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import multiprocessing as mp

from dia_common import DiaRun, parameters
from savetimspy.get_frames import write_frames
from savetimspy.get_diagonals import write_diagonals

source = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
target = pathlib.Path("tests/diagonals")

compression_level = 1
make_all_frames_seem_unfragmented = True
processesNo = 5

target_paths = write_diagonals(source, target, processesNo, compression_level, make_all_frames_seem_unfragmented)

