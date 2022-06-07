import pathlib

from collections import namedtuple
from opentimspy import OpenTIMS
from typing import Iterable

from savetimspy import SaveTIMS
from savetimspy.common_assertions import (
    assert_minimal_input_for_clusterings_exist,
)


FrameDataset = namedtuple(
    "FrameDataset", 
    "total_scans src_frame df"
)


def write_frame_datasets(
    frame_datasets: Iterable[FrameDataset],
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
                    scans=frame_dataset.df.scan.to_numpy(),
                    tofs=frame_dataset.df.tof.to_numpy(),
                    intensities=frame_dataset.df.intensity.to_numpy(),
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

