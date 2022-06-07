%load_ext autoreload
%autoreload 2
from savetimspy.get_frames import write_frames
from savetimspy.get_ms1 import write_ms1
from savetimspy.get_ms2 import write_ms2, combined_ms2_frames_generator
from savetimspy.get_diagonals import write_diagonals
from savetimspy.get_hprs import (
    write_hprs,
    make_overlapping_HPR_mz_intervals,
)

from MSclusterparser.parser import read_4DFF_to_df_physical_dims_only
from py4DFF import Run4DFFv4_12_1
from tqdm import tqdm
from MSclusterparser.boxes_ops import *
import matplotlib.pyplot as plt
import pathlib
# from midiapipe.write_diagonals

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")
fragHeLa = _get_d("*3343.d")
unfragHeLa = _get_d("*3342.d")

# source = unfrag5P
# source = unfragHeLa
# source = fragHeLa

HPR_intervals = make_overlapping_HPR_mz_intervals(
    min_mz=300.0,
    max_mz=1_500.0,
)


for source in (frag5P, unfrag5P, fragHeLa, unfragHeLa):
    ms1_faster_d = write_ms1(source=source, target=source/"ms1_faster.d", verbose=True)
    ms2_faster_d = write_ms2(source=source, target=source/"ms2_faster.d", verbose=True)
    diagonals = write_diagonals(source=source, target=source/"diagonals_faster.d", verbose=True)
    hprs = write_hprs(
        HPR_intervals=HPR_intervals,
        source=source,
        target=source/"hprs_faster",
        verbose=True,
    )

# Run4DFFv4_12_1(ms1_faster_d)
# ms1_faster_features = read_4DFF_to_df_physical_dims_only(Run4DFFv4_12_1(ms1_faster_d))
# ms1_slo_incorrect_rts_features = read_4DFF_to_df_physical_dims_only(Run4DFFv4_12_1(source/"ms1.d"))
# # ms1_fixed_rts_features = read_4DFF_to_df_physical_dims_only(Run4DFFv4_12_1(source/"ms1_fixed_rts.d"))
# ms1_vanilla_features = read_4DFF_to_df_physical_dims_only(Run4DFFv4_12_1(source)) 

# ms1_vanilla_features
# ms1_faster_features
# # ms1_fixed_rts_features

