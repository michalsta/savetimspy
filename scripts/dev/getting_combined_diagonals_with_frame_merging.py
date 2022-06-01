%load_ext autoreload
%autoreload 2
from savetimspy.get_frames import write_frames
from savetimspy.get_ms1 import write_ms1
from savetimspy.get_ms2 import write_ms2, combined_ms2_frames_generator
from savetimspy.get_diagonals import write_diagonals
from savetimspy.get_hprs import write_hprs
from savetimspy.write_frame_datasets import write_frame_datasets
import shutil
import pathlib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import sqlite3
import opentimspy
import matplotlib.pyplot as plt

from MSclusterparser.boxes_ops import *
from py4DFF import Run4DFFv4_12_1
from MSclusterparser.parser import read_4DFF_to_df_physical_dims_only
write_frame_datasets

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")

source = unfrag5P
target = project_folder/"tests"/source.name

write_frame_datasets(
    frame_datasets=combined_ms2_frames_generator(source, verbose=True),
    source=source,
    target=target,
    set_MsMsType_to_0=True,
    run_deduplication=False,
)
features = Run4DFFv4_12_1(target)
clusters = read_4DFF_to_df_physical_dims_only(features)
clusters = get_extents_vol_centers(clusters)

extent_cols = ["mz_extent","inv_ion_mobility_extent","retention_time_extent"]
pd.plotting.scatter_matrix(
    clusters.query("retention_time_extent < 55")[extent_cols],
    hist_kwds={"bins":101},
    grid=True,
    s=1
)
plt.show()


_get_clusters = lambda p: read_4DFF_to_df_physical_dims_only(p/f"{p.name}.features")
ms1_vanilla = _get_clusters(source)
ms1_vanilla = get_extents_vol_centers(ms1_vanilla)
# shutil.rmtree(target)
# target.exists()

pd.plotting.scatter_matrix(
    ms1_vanilla.query("retention_time_extent < 55")[extent_cols],
    hist_kwds={"bins":101},
    grid=True,
    s=1
)
plt.show()


# import plotnine as p

# df0 = clusters[extent_cols].copy()
# df0["experiment"] = "unfragmented 5P combined MS2 deduplicated with MS1 RTs"
# df1 = ms1_vanilla[extent_cols].copy()
# df1["experiment"] = "unfragmented 5P Vanilla MS1"
# df = pd.concat([df0, df1], ignore_index=True)
# df = df.query("mz_extent < 0.1 and mz_extent > 0")

# df_long = pd.melt(df, id_vars=("experiment"), var_name="dimension")
# (   
#     p.ggplot(df_long) + 
#     p.geom_histogram(p.aes(x="value")) +
#     p.theme(legend_position="top") +
#     p.facet_grid("experiment ~ dimension", scales="free_x")
# )


