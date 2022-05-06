%load_ext autoreload
%autoreload 2

"""This all can be optimized later on by specific uses of savetimspy."""
from savetimspy.specific_writers import write_diagonals
import pathlib

from opentimspy.sql import table2dict

source = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d")
target = pathlib.Path("data/raw/G211125_007_Slot1-1_1_3264.d/diagonals")
# target_paths = write_diagonals(source, target, 10)

window_group, window_frames, source, target = next(_input_stream)

get_groups_as_consecutive_ints(df.frame.values)

df.frame.unique()

np.all(window_frames.Frame.values[1:] - window_frames.Frame.values[:-1] > 0)
np.diff(window_frames, )

window_frames

frame_to_original_frame[1]
def trivial_group_index_map(xx: np.array):
    unique_xx = np.unique(xx)    
    return dict(enumerate(unique_xx, start=1))

pd.DataFrame(rawdata.query(
        frames=1,
        columns="frame scan tof intensity".split()))


frame_to_original_frame[1]