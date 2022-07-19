import numpy as np
import pathlib
import typing

from collections import defaultdict


def assert_minimal_input_for_clusterings_exist(path: pathlib.Path):
    assert path.exists(), f"File {path} does not exist."
    tdf = path / "analysis.tdf"
    assert tdf.exists(), f"File {tdf} does not exist."
    tdf_bin = path / "analysis.tdf_bin"
    assert tdf_bin.exists(), f"File {tdf_bin} does not exist."


def assert_data_the_same(old, new, columns=("scan", "tof", "intensity")):
    for col in columns:
        assert np.all(old[col] == new[col]), f"{col} differ"


def iter_differences_to_closest_mzs(
    cluster_of_events_from_vanilla_4DFF, hprs, hpr_idx
) -> typing.Iterator[float]:
    for (frame, cycle, step), data in cluster_of_events_from_vanilla_4DFF.groupby(
        ["frame", "cycle", "step"]
    ):
        hpr_data = hprs.get_hpr_cycle_step_data(
            hpr_idx=hpr_idx,
            cycle=cycle,
            step=step,
            columns=("scan", "mz", "intensity"),
        )
        scan_intesity_to_mzs = defaultdict(list)
        for scan, mz, intensity in zip(*hpr_data.values()):
            scan_intesity_to_mzs[(scan, intensity)].append(mz)
        scan_intesity_to_mzs = {
            scan_intesity: np.array(mzs)
            for scan_intesity, mzs in scan_intesity_to_mzs.items()
        }
        for data_scan, data_intensity, data_mz in zip(
            data.scan, data.intensity, data.mz
        ):
            mzs = scan_intesity_to_mzs[(data_scan, data_intensity)]
            min_idx = np.argmin(np.abs(mzs - data_mz))
            yield mzs[min_idx] - data_mz
