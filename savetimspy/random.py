import numpy as np
import pandas as pd


def get_random_uint32(size: int, _low: int=1, _high: int=1_000) -> np.array:
    return np.random.randint(size=size, low=_low, high=_high, dtype=np.uint32)

def get_random_frames(size: int, _low: int=1, _high: int=1_000) -> np.array:
    return get_random_uint32(size,_low,_high)

def get_random_scans(size: int, _low: int=1, _high: int=900) -> np.array:
    return get_random_uint32(size,_low,_high)

def get_random_tofs(size: int, _low: int=3_000, _high: int=40_000) -> np.array:
    return get_random_uint32(size,_low,_high)

def get_random_intensities(size: int, _low: int=1, _high: int=10_000) -> np.array:
    return get_random_uint32(size,_low,_high)

def get_random_df(size: int) -> pd.DataFrame:
    """Make random data-frame."""
    return pd.DataFrame({
        "frame": get_random_frames(size),
        "scan": get_random_scans(size),
        "tof": get_random_tofs(size),
        "intensity": get_random_intensities(size),
    })
