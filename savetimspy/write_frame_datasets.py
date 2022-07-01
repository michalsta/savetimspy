import numpy as np
import numpy.typing as npt
import pathlib

from collections import namedtuple
from dataclasses import dataclass
from opentimspy import OpenTIMS
from typing import Iterable

from savetimspy import SaveTIMS
from savetimspy.common_assertions import (
    assert_minimal_input_for_clusterings_exist,
)


# this is not very performant, but that is not important.
# I simply want to have type annotation.
@dataclass
class FrameSaveBundle:
    """
    A base unit of information for saving one frame.
    """
    total_scans: int
    src_frame: int
    scans: npt.NDArray[np.uint32]
    tofs: npt.NDArray[np.uint32]
    intensities: npt.NDArray[np.uint32]



def write_frame_datasets(
    frame_datasets: Iterable[FrameSaveBundle],
    source: pathlib.Path,
    target: pathlib.Path,
    compression_level: int=1,
    set_MsMsType_to_0: bool=False,
    run_deduplication: bool=True,
    verbose: bool=False,
) -> pathlib.Path:
    """
    A simple wrapper around savetimspy.
    """
    try:
        if verbose:
            from tqdm import tqdm
            frame_datasets = tqdm(frame_datasets)

        with OpenTIMS(source) as ot,\
             SaveTIMS(ot, target, compression_level) as saviour:
            for frame_dataset in frame_datasets:
                kwargs = {}
                if set_MsMsType_to_0:
                    kwargs["MsMsType"] = 0
                saviour.save_frame_tofs(
                    scans=frame_dataset.scan,
                    tofs=frame_dataset.tof,
                    intensities=frame_dataset.intensity,
                    total_scans=frame_dataset.total_scans,
                    src_frame=frame_dataset.src_frame,
                    run_deduplication=run_deduplication,
                    **kwargs
                )

        if verbose:
            print(f"Finished with: {target}")

    except FileExistsError:
        assert_minimal_input_for_clusterings_exist(target)
        if verbose:
            print(f"Results already there ({target}): not repeating.")        

    return target 

