import numpy as np
import pandas as pd


def get_random_uint32(size: int, _low: int=1, _high: int=1_000, **kwargs) -> np.array:
    return np.random.randint(size=size, low=_low, high=_high, dtype=np.uint32)

def get_random_frames(size: int, min_frame: int=1, max_frame: int=1_000, **kwargs) -> np.array:
    return get_random_uint32(size, min_frame, max_frame)

def get_random_scans(size: int, min_scan: int=1, max_scan: int=900, **kwargs) -> np.array:
    return get_random_uint32(size, min_scan, max_scan)

def get_random_tofs(size: int, min_tof: int=3_000, max_tof: int=40_000, **kwargs) -> np.array:
    return get_random_uint32(size, min_tof, max_tof)

def get_random_intensities(size: int, min_intensity: int=1, max_intensity: int=10_000, **kwargs) -> np.array:
    return get_random_uint32(size, min_intensity, max_intensity)

def get_random_df(size: int, **kwargs) -> pd.DataFrame:
    """Make random data-frame."""
    return pd.DataFrame({
        "frame": get_random_frames(size, **kwargs),
        "scan": get_random_scans(size, **kwargs),
        "tof": get_random_tofs(size, **kwargs),
        "intensity": get_random_intensities(size, **kwargs),
    })
