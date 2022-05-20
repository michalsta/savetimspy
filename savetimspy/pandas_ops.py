import numba
import numpy as np
import pandas as pd

from typing import Any, List, Iterator, Tuple

def deduplicate(
    df: pd.DataFrame,
    key_columns: List[str]="frame scan tof".split(),
    sort: bool=True,
) -> pd.DataFrame:
    """Deduplicate and sort the data frame using a given key columns."""
    for key_column in key_columns:
        assert key_column in df.columns, f"Missing column '{key_column}'."
    assert "intensity" in df.columns, "Missing column 'intensity'."
    return df.groupby(
        key_columns,
        sort=sort,
        as_index=False,
    ).intensity.sum()


# def is_sorted(xx: np.array) -> bool:
#     return np.all(xx[:-1] <= xx[1:])

@numba.jit
def is_sorted(xx):
    x_prev = xx[0]
    for x in xx:
        if x < x_prev:
            return False
    return True



def iter_group_based_views_of_data(
    df: pd.DataFrame,
    grouping_column_name: str,
    assert_sorted: bool=True,
) -> Iterator[Tuple[Any, pd.DataFrame]]:
    """Iterate over frame-based views of the dataframe."""
    assert grouping_column_name in df.columns, f"Lacking column {grouping_column_name} in the submitted 'df'."
    groups = df[grouping_column_name].values
    if assert_sorted:
        assert is_sorted(groups), "The df needs to be sorted with respect to frames for 'iter_frame_based_views' to iterate over views of data properly without any copies."
    i_prev = 0
    group_prev = groups[0]
    for i, group in enumerate(groups):
        if group != group_prev:
            sub_df = df.iloc[i_prev:i]
            yield int(group_prev), sub_df
            i_prev = i
            group_prev = group
    if i_prev <= i:
        sub_df = df.iloc[i_prev:]
        if len(sub_df):
            yield int(group_prev), sub_df

# # deduplication does the same thing as savetimspy.numba_helper.deduplicate
# from savetimspy.numba_helper import deduplicate as deduplicate_numba
# df1 = df.query('frame == 1')

# w = deduplicate_numba(
#     scans=df1.scan.values,
#     tofs=df1.tof.values,
#     intensities=df1.intensity.values)
# w = pd.DataFrame(np.vstack(w).T, columns='scan tof intensity'.split())
# (w == deduplicate(df1).iloc[:,1:]).all()

def trivial_group_index_map(xx: np.array):
    unique_xx = np.unique(xx)    
    return dict(enumerate(unique_xx, start=1))
