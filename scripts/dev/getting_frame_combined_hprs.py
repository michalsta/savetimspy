%load_ext autoreload
%autoreload 2
%load_ext snakeviz
from savetimspy.get_frames import write_frames
from savetimspy.get_ms1 import write_ms1
from savetimspy.get_ms2 import write_ms2, combined_ms2_frames_generator
from savetimspy.get_diagonals import write_diagonals
from savetimspy.get_hprs import (
    write_hprs,
    make_overlapping_HPR_mz_intervals,
    HPRS,
    HPRbyte,
    write_hprs,
)
from savetimspy.write_frame_datasets import write_frame_datasets
import shutil
import pathlib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 43)
import sqlite3
import opentimspy

from dia_common import DiaRun
from MSclusterparser.parser import read_4DFF_to_df_physical_dims_only
from py4DFF import Run4DFFv4_12_1
from savetimspy.write_frame_datasets import (
    write_frame_datasets,
    FrameDataset
)
from tqdm import tqdm
from MSclusterparser.boxes_ops import *
from clusterMS.plotting import scatterplot_matrix_alla_polacca

import matplotlib.pyplot as plt
import plotnine as p
import typing
import matplotlib.style as mplstyle

project_folder = pathlib.Path(".").expanduser()
rawdata_folder = project_folder/"rawdata"

_get_d = lambda x: next(rawdata_folder.glob(x))
frag5P = _get_d("*3516.d") 
unfrag5P = _get_d("*3517.d")
fragHeLa = _get_d("*3343.d")
unfragHeLa = _get_d("*3342.d")

# source = unfrag5P
# source = unfragHeLa
source = fragHeLa
target = source/"hprs_faster"

HPR_intervals = make_overlapping_HPR_mz_intervals(
    min_mz=300.0,
    max_mz=1_500.0,
)

# %%snakeviz
hpr_folders = write_hprs(
    HPR_intervals=HPR_intervals,
    source=source,
    target=target,
    combine_steps_per_cycle=True,
    min_coverage_fraction=0.5,# no unit
    window_width=36.0,# Da
    min_scan=1,
    max_scan=None,
    compression_level=1,
    make_all_frames_seem_unfragmented=True,
    verbose=True,
    # _max_iterations=10_000,
)

# feature_folders = [Run4DFFv4_12_1(hpr_d, verbose=True) for hpr_d in hpr_folders]
# points = [len(opentimspy.OpenTIMS(hpr_folder)) for hpr_folder in hpr_folders]
# HPR_intervals['center'] = HPR_intervals.eval("(hpr_start + hpr_stop)/2.0")
# plt.plot(HPR_intervals.center, points)
# plt.scatter(HPR_intervals.center, points)
# plt.xlabel("Center of the HPR")
# plt.ylabel("raw events count")
# plt.show()



# hpr_clusters = []
# for hpr_idx, ff in tqdm(enumerate(feature_folders)):
#     hpr = read_4DFF_to_df_physical_dims_only(ff)
#     hpr = get_extents_vol_centers(hpr)
#     hpr["experiment"] = hpr_idx
#     hpr_clusters.append(hpr)

# extent_cols = ["mz_extent","inv_ion_mobility_extent","retention_time_extent"]
# pd.plotting.scatter_matrix(
#     hpr_clusters[np.argmax(points)].query("retention_time_extent < 55")[extent_cols],
#     hist_kwds={"bins":101},
#     grid=True,
#     s=1
# )
# plt.suptitle(f"Feature Extents For Most Event-Rich ({max(points):_} events) HPR No {np.argmax(points)}.")
# plt.show()

# pd.plotting.scatter_matrix(
#     hpr_clusters[np.argmin(points)].query("retention_time_extent < 55")[extent_cols],
#     hist_kwds={"bins":101},
#     grid=True,
#     s=1
# )
# plt.suptitle(f"Feature Extents For Least Event-Rich ({min(points):_} events) HPR No {np.argmin(points)}.")
# plt.show()

# hpr_clusters_df = pd.concat(hpr_clusters, ignore_index=True)
# hpr_clusters_df.query("vol>0")
# MS1 = read_4DFF_to_df_physical_dims_only(source/f"{source.name}.features")
# MS1 = get_extents_vol_centers(MS1)

# hpr_clusters_df.to_hdf(source/"features.hdf", "hprs")
# MS1.to_hdf(source/"features.hdf", "ms1")

hpr_clusters_df = pd.read_hdf(source/"features.hdf", "hprs")
MS1 = pd.read_hdf(source/"features.hdf", "ms1")
MS1 = get_extents_vol_centers(MS1)
# pd.plotting.scatter_matrix(
#     MS1.query("retention_time_extent < 55")[extent_cols],
#     hist_kwds={"bins":101},
#     grid=True,
#     s=1
# )
# plt.suptitle("MS1 vanilla 4DFF")
# plt.show()

hpr_sizes = hpr_clusters_df.groupby("experiment").size()
plt.hist(MS1.mz_extent, bins=1001, label="MS1")
plt.hist(hpr_clusters[np.argmax(points)].mz_extent, bins=1001, label="HPR")
plt.legend()
plt.show()

plt.scatter(MS1.mz_center, MS1.inv_ion_mobility_center, s=.5, alpha=.5)
plt.show()


mplstyle.use(['fast'])

plt.scatter(MS1.mz_center,
            MS1.inv_ion_mobility_center,
            alpha=normalize(MS1.intensity),
            s=4)
plt.plot([250, 500,1_000], [0.6, 1, 1.4])
plt.show()

def point_to_intercept_and_slope(x0,y0,x1,y1):
    slope = (y1-y0)/(x1-x0)
    intercept = y0 - slope*x0
    return intercept, slope

b0,a0 = point_to_intercept_and_slope(x0=250, y0=0.6, x1=500, y1=1.0)
b1,a1 = point_to_intercept_and_slope(x0=500, y0=1.0, x1=1_000, y1=1.4)
MS1["charge_1"] = ~MS1.eval(f"{a0}*mz_center + {b0} >= inv_ion_mobility_center and {a1}*mz_center + {b1} >= inv_ion_mobility_center")
    
plt.scatter(MS1.mz_center,
            MS1.inv_ion_mobility_center,
            alpha=normalize(MS1.intensity),
            s=4,
            c=np.where(MS1.charge_1,'blue','orange'))
plt.plot([250, 500,1_000], [0.6, 1, 1.4])
plt.show()


def get_quantiles(x, bins=1_000):
    xx = np.linspace(0,1,bins+1)
    return pd.DataFrame({"prob": xx, "quantile": np.quantile(x, xx)})


MS1_quant = get_quantiles(MS1.mz_extent)
MS1_quant["exp"] = "MS1"
HPR_quant = get_quantiles(hpr_clusters[np.argmax(points)].mz_extent)
HPR_quant["exp"] = f"HPR No {np.argmax(points)}"

mz_extents = pd.concat([MS1_quant, HPR_quant], ignore_index=True)
( p.ggplot(mz_extents) + p.geom_line(p.aes(x="quant", y="prob", color="exp")) )



pd.plotting.scatter_matrix(
    MS1.query("retention_time_extent < 55 and ~charge_1")[extent_cols],
    hist_kwds={"bins":101},
    grid=True,
    s=1
)
plt.suptitle(f"Feature Extents For MS1 Multicharged Region")
plt.show()

extent_cols = ["mz_extent","inv_ion_mobility_extent","retention_time_extent"]
MS1_nonzero = MS1[np.any(MS1[extent_cols] > 0, axis=1)]
for c in extent_cols:
    MS1_nonzero
X = pd.DataFrame({c:pd.qcut(x=MS1_nonzero[c], q=[0, 0.05, 0.95, 1.0], labels=['small','good','large'])
    for c in extent_cols} 
)
MS1_90_perc = MS1_nonzero[np.all(X == 'good',axis=1)].copy()

MS1_90_perc["intensity_over_vol"] = MS1_90_perc.intensity / MS1_90_perc.vol
MS1_90_perc["log10_intensity_over_vol"] = np.log10(MS1_90_perc.intensity_over_vol)
MS1_90_perc["log10_intensity"] = np.log10(MS1_90_perc.intensity)
pd.plotting.scatter_matrix(
    MS1_90_perc[[
        "mz_extent",
        "inv_ion_mobility_extent",
        "retention_time_extent",
        "log10_intensity",
        "log10_intensity_over_vol"
    ]],
    hist_kwds={"bins":101},
    grid=True,
    s=1
)
plt.suptitle(f"Feature Extents For MS1 Multicharged Region: 5-95 centiles")
plt.show()

columns = extent_cols


MS1_multicharge = MS1.query('~charge_1 and vol>0').copy()
column_to_min_max_tuple = column_quantiles(MS1_multicharge[extent_cols])
MS1_final = cut_df(MS1_multicharge, **column_quantiles(MS1_multicharge[extent_cols])).copy()
MS1_final["intensity_over_vol"] = MS1_final.intensity / MS1_final.vol

# where to put these pandas ops?
# add in some meaningful matplotlib visualization.
# diagonal: histograms
# off diagonal: heatmaps after some binning

plt.hexbin(MS1_final.mz_extent, MS1_final.intensity, yscale='log')
plt.show()

columns = ['mz_extent', 'inv_ion_mobility_extent', 'retention_time_extent', 'intensity', 'intensity_over_vol']


with plt.style.context('dark_background'):
    scatterplot_matrix_alla_polacca(
        df=MS1_final[columns],
        hexbin_kwargs={"cmap":"inferno"},
        log_scale={"intensity","intensity_over_vol"},
    )

hprs_preprocessed = hpr_clusters_df.query("vol>0")
hprs_preprocessed = cut_df(hprs_preprocessed, **column_quantiles(hprs_preprocessed[extent_cols])).copy()
hprs_preprocessed["intensity_over_vol"] = hprs_preprocessed.intensity / hprs_preprocessed.vol

with plt.style.context('dark_background'):
    scatterplot_matrix_alla_polacca(
        df=hprs_preprocessed[columns],
        hexbin_kwargs={"cmap":"inferno"},
        log_scale={"intensity","intensity_over_vol"}
    )
# moglibyśmy to wszystko jeszcze zdyskretyzować, albo, przynajmniej, zdyskredytować.


