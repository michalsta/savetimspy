from __future__ import annotations
from tqdm import tqdm

import cmath
import collections
import functools
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import pathlib
import sqlite3
import typing


from savetimspy.common_assertions import (
    assert_minimal_input_for_clusterings_exist,
)
from savetimspy.fs_ops import get_limits, set_soft_limit
from savetimspy.interval_ops import NestedContainmentList
from savetimspy.numba_helper import (
    coordinatewise_range,
    binary_search,
    dedup_v2,
    get_group_tags_starts_ends,
)
from savetimspy.pandas_ops import is_sorted
from dia_common.dia_main import DiaRun
from savetimspy.writer import SaveTIMS
from savetimspy.write_frame_datasets import FrameSaveBundle

SCAN_TOF_INTENSITY_ARRAYS_DICT_np_uint32 = dict[str, npt.NDArray[np.uint32]]


def make_overlapping_HPR_mz_intervals(
    min_mz: float = 300.0,
    max_mz: float = 1500.0,
    width: float = 12.0,
    add_overlaps: bool = True,
) -> pd.DataFrame:
    mz_borders = []
    mz_border = min_mz
    while mz_border <= max_mz:
        mz_borders.append(mz_border)
        mz_border += width
    mz_borders = np.array(mz_borders)
    HPR_intervals = pd.DataFrame(
        {
            "hpr_start": mz_borders[:-1],
            "hpr_stop": mz_borders[1:],
        }
    )
    if add_overlaps:
        HPR_intervals = pd.concat((HPR_intervals, (HPR_intervals + 6).iloc[:-1]))
        HPR_intervals = HPR_intervals.sort_values(["hpr_start", "hpr_stop"])
        HPR_intervals.index = range(len(HPR_intervals))
    HPR_intervals.index.name = "hpr_idx"
    return HPR_intervals


def composed_of_consecutive_integers(xx: npt.NDArray[int]) -> bool:
    return np.max(xx) - np.min(xx) + 1 == len(xx)


def infer_the_max_usable_scan(DiaFrameMsMsWindows: pd.DataFrame) -> int:
    """Infer the maximal usable scan.

    After each fragmentation the quadrupole needs few last scans to be reset to the proper position.
    This does not seem to be the case for the last frame.
    Data collected during this resetting is weird and we neglect it.
    """
    return int(DiaFrameMsMsWindows.groupby("step").ScanNumBegin.max()[:-1].min())


def combine_hpr_step_datasets(
    hpr_step_datasets: list[SCAN_TOF_INTENSITY_ARRAYS_DICT_np_uint32],
) -> SCAN_TOF_INTENSITY_ARRAYS_DICT_np_uint32:
    points_cnt = sum(len(dataset["scan"]) for dataset in hpr_step_datasets)
    all_scans = np.empty(shape=(points_cnt,), dtype=np.uint32)
    all_tofs = np.empty(shape=(points_cnt,), dtype=np.uint32)
    all_intensities = np.empty(shape=(points_cnt,), dtype=np.uint32)
    i = 0
    for dataset in hpr_step_datasets:
        n = len(dataset["scan"])
        all_scans[i : i + n] = dataset["scan"]
        all_tofs[i : i + n] = dataset["tof"]
        all_intensities[i : i + n] = dataset["intensity"]
        i += n
    return (all_scans, all_tofs, all_intensities)


MEASUREMENTS_TYPE = dict[
    str,  # column name
    npt.NDArray[np.uint32] | npt.NDArray[np.float64],  # measurements
]

# TODO: move away from forcing scans into a smaller matrix, as if they started from 0 not 1.
class HPRS:
    def __init__(
        self,
        HPR_intervals: pd.DataFrame,
        dia_run: DiaRun,
        min_coverage_fraction: float = 0.5,
        window_width: float | None = 36.0,
        min_scan: int | None = 1,
        max_scan: int | None = None,
        right_quadrupole_buffer: float = 0.01,
        hpr_idx_scan_to_max_steps: int = 3,  # TODO: and inference would be better than a hardcoded value.
    ):
        self.HPR_intervals = HPR_intervals
        self.hpr_indices = HPR_intervals.index.to_numpy()
        self.dia_run = dia_run
        self.min_coverage_fraction = min_coverage_fraction
        self.window_width = window_width
        # defining quadrupole positions
        self.DiaFrameMsMsWindows = self.dia_run.DiaFrameMsMsWindows
        self.steps = self.DiaFrameMsMsWindows.step.unique()
        if self.window_width is None:
            quad_borders_exp = """
                quadrupole_start = IsolationMz - IsolationWidth / 2.0
                quadrupole_stop  = IsolationMz + IsolationWidth / 2.0 - @right_quadrupole_buffer
            """
        else:
            quad_borders_exp = """
                quadrupole_start = IsolationMz - @self.window_width / 2.0
                quadrupole_stop  = IsolationMz + @self.window_width / 2.0 - @right_quadrupole_buffer
            """
        self.DiaFrameMsMsWindows = self.DiaFrameMsMsWindows.eval(quad_borders_exp)

        # defining scans of interest
        self.min_scan = min_scan
        if max_scan is None:
            self.max_scan = infer_the_max_usable_scan(self.DiaFrameMsMsWindows)

        self.scans = np.arange(self.min_scan, self.max_scan + 1)
        # Mapping (step,scan) to actual quadrupole positions

        self.StepScanToQuadrupole = (
            self.DiaFrameMsMsWindows.loc[
                np.repeat(
                    self.DiaFrameMsMsWindows.index,
                    self.DiaFrameMsMsWindows.eval("ScanNumEnd - ScanNumBegin"),
                ),  # repeats the index the total number of scan times
                [  # result must contain:
                    "step",
                    "quadrupole_start",
                    "quadrupole_stop",
                    "CollisionEnergy",
                    "IsolationMz",
                ],
            ]
            .reset_index()
            .rename(columns={"index": "quadrupole_idx"})
        )

        self.StepScanToQuadrupole["scan"] = coordinatewise_range(
            # silly convention: scans start from 1
            self.DiaFrameMsMsWindows.ScanNumBegin.to_numpy() + 1,
            self.DiaFrameMsMsWindows.ScanNumEnd.to_numpy() + 1,
        )
        self.StepScanToQuadrupole = self.StepScanToQuadrupole.query(
            "@self.min_scan <= scan and scan <= @self.max_scan"
        ).reset_index(drop=True)

        self.StepScanToQuadrupolePosition = (
            self.StepScanToQuadrupole.pivot(
                index="scan",
                columns="step",
                values="quadrupole_idx",
            )
            .fillna(-1)
            .astype(int)
        )
        # convention: -1 above = no valid quadrupole position mapped

        scans_below_min_above_max = set(
            self.StepScanToQuadrupole.query(
                f"scan < {self.min_scan} or scan > {self.max_scan}"
            ).quadrupole_idx
        )
        self.StepScanToQuadrupolePosition[
            self.StepScanToQuadrupolePosition.isin(scans_below_min_above_max)
        ] = -1
        self.StepScanToQuadrupole = self.StepScanToQuadrupole[
            ~self.StepScanToQuadrupole.scan.isin(scans_below_min_above_max)
        ]

        # mapping hypothetical precursor ranges to quadrupole positions
        #   * initially every intersecting quadrupole position is valid
        #   * shortlisting quadrupole positions that intersect with the quadrupole above a given percentage of length.
        interval_db = NestedContainmentList.from_df(HPR_intervals)

        # these are indices of self.StepScanToQuadrupole, not quadrupole_idx: good
        StepScanToQuadrupole_idxs, hpr_idxs = interval_db.query_df(
            self.StepScanToQuadrupole[["quadrupole_start", "quadrupole_stop"]]
        )

        self.hpr_quadrupole_matches = pd.concat(
            [
                self.StepScanToQuadrupole.loc[StepScanToQuadrupole_idxs].reset_index(),
                self.HPR_intervals.loc[hpr_idxs].reset_index(),
            ],
            axis=1,
        )

        for (hpr_idx, step), scans in self.hpr_quadrupole_matches.groupby(
            ["hpr_idx", "step"]
        ).scan:
            assert (
                len(scans) == scans.max() - scans.min() + 1
            ), f"Steps do not form a consecutive integer sequence in case of hpr_idx={hpr_idx} and step={step}"

        self.hpr_quadrupole_matches["quadrupole_coverage_dalton"] = np.minimum(
            self.hpr_quadrupole_matches.hpr_stop,
            self.hpr_quadrupole_matches.quadrupole_stop,
        ) - np.maximum(
            self.hpr_quadrupole_matches.hpr_start,
            self.hpr_quadrupole_matches.quadrupole_start,
        )

        self.hpr_quadrupole_matches[
            "quadrupole_coverage_fraction"
        ] = self.hpr_quadrupole_matches.quadrupole_coverage_dalton / (
            self.hpr_quadrupole_matches.hpr_stop - self.hpr_quadrupole_matches.hpr_start
        )

        self.hpr_quadrupole_matches = self.hpr_quadrupole_matches.query(
            "quadrupole_coverage_fraction >= @self.min_coverage_fraction"
        ).copy()

        self.hpr_quadrupole_matches = self.hpr_quadrupole_matches.reset_index(
            drop=True
        )  # don't need that old index

        # Quality Check: with a given width of window, at most k steps should intersect with our data:
        hpr_idx_scan_to_steps = self.hpr_quadrupole_matches.groupby(
            ["hpr_idx", "scan"]
        ).step.size()
        for (hpr_idx, scan), steps_cnt in hpr_idx_scan_to_steps.iteritems():
            assert (
                steps_cnt <= hpr_idx_scan_to_max_steps
            ), f"The actual number of steps per scan per hpr_idx={hpr_idx} and scan={scan} was {steps_cnt} exceeding the expected value of {hpr_idx_scan_to_max_steps}."

        # Setting some step, scan positions to unuseful (-1):
        self.StepScanToUsedQuadrupolePosition = self.StepScanToQuadrupolePosition.copy()
        self.StepScanToUsedQuadrupolePosition[
            ~self.StepScanToUsedQuadrupolePosition.isin(
                self.hpr_quadrupole_matches.quadrupole_idx.unique()
            )
        ] = -1
        self.UnusedStepScanPositionsCount = sum(
            self.StepScanToUsedQuadrupolePosition.values == -1
        ).sum()

        # Making a list of data frames that each map scan to hpr_idx
        # this is used in the iterator to map hpr_idx back unto the events.
        empty_df = pd.DataFrame(columns=["hpr_idx"])
        empty_df.index.name = "scan"
        self.step_to_scan_hpr_dfs = [empty_df] * (
            self.dia_run.max_step + 1
        )  # List must have all scan numbers
        for step, data in self.hpr_quadrupole_matches.groupby("step")[
            ["scan", "hpr_idx"]
        ]:
            data["scan"] = data.scan.astype(np.uint32)
            self.step_to_scan_hpr_dfs[step] = data.set_index("scan")

        for step, step_to_scan_hpr_df in enumerate(self.step_to_scan_hpr_dfs):
            if len(
                step_to_scan_hpr_df
            ):  # some lists can be empty: but we need them anyway for the indexing.
                assert is_sorted(
                    step_to_scan_hpr_df.index.values
                ), f"Step's {step} step_to_scan_hpr_df scan index ain't sorted."

        self.hpr_step_to_scan_min_max = {
            (hpr_idx, step): (np.min(scans), np.max(scans))
            for (hpr_idx, step), scans in self.hpr_quadrupole_matches.groupby(
                ["hpr_idx", "step"]
            ).scan
        }
        # Below: the empty data with all possible column names:
        # automatically check for proper column name
        self.empty_data = {}
        for col in ("frame", "scan", "tof", "intensity"):
            self.empty_data[col] = np.array([], dtype=np.uint32)
        for col in ("retention_time", "inv_ion_mobility", "mz"):
            self.empty_data[col] = np.array([], dtype=np.float64)
        # TODO: generalize to multiple diagonals
        self.diagonals = (1, 2, 3)

    @property
    def min_step(self):
        return self.dia_run.min_step

    @property
    def max_step(self):
        return self.dia_run.max_step

    @property
    def scan_cnt(self):
        return self.max_scan - self.min_scan + 1

    @property
    def step_cnt(self):
        return self.dia_run.step_cnt

    def hpr_idx_scan_to_step(self) -> dict[tuple[int], tuple[int]]:
        return {
            k: tuple(v)
            for k, v in self.hpr_quadrupole_matches.groupby(["hpr_idx", "scan"]).step
        }

    @property
    def diagonal_cnt(self):
        return len(self.diagonals)

    # TODO: speed this up
    # TODO: rename this to scan_step_to_diagonal
    @functools.cache
    def get_subframe_matrix(self, hpr_idx) -> npt.NDArray[int]:
        hpr_quad_matches = self.hpr_quadrupole_matches.query(
            "hpr_idx == @hpr_idx"
        ).copy()
        hpr_idxs_touching_the_quadrupole_position = set(hpr_quad_matches.quadrupole_idx)
        scan_step_sizes = hpr_quad_matches.groupby("scan").step.size()
        min_steps_quads = set()
        max_steps_quads = set()
        scans_with_most_steps = scan_step_sizes.index[
            scan_step_sizes == max(scan_step_sizes)
        ]
        assert composed_of_consecutive_integers(
            scans_with_most_steps
        ), f"The set of non-edge cases is not connected: there are some holes in the seqeuence of scans that each cover the same number of steps, i.e. {max(scan_step_sizes)}"
        for scan, df in hpr_quad_matches.query(
            "scan in @scans_with_most_steps"
        ).groupby("scan")[["step", "quadrupole_idx"]]:
            quad_idx_of_min_step = df.quadrupole_idx.iloc[df.step.argmin()]
            quad_idx_of_max_step = df.quadrupole_idx.iloc[df.step.argmax()]
            min_steps_quads.add(quad_idx_of_min_step)
            max_steps_quads.add(quad_idx_of_max_step)
        scans_without_smallest_step = set(
            range(hpr_quad_matches.scan.min(), scans_with_most_steps.min())
        )
        for scan, df in hpr_quad_matches.query(
            "scan in @scans_without_smallest_step"
        ).groupby("scan")[["step", "quadrupole_idx"]]:
            quad_idx_of_max_step = df.quadrupole_idx.iloc[df.step.argmax()]
            max_steps_quads.add(quad_idx_of_max_step)
        scans_without_biggest_step = set(
            range(scans_with_most_steps.max(), hpr_quad_matches.scan.max() + 1)
        )
        for scan, df in hpr_quad_matches.query(
            "scan in @scans_without_biggest_step"
        ).groupby("scan")[["step", "quadrupole_idx"]]:
            quad_idx_of_min_step = df.quadrupole_idx.iloc[df.step.argmin()]
            min_steps_quads.add(quad_idx_of_min_step)
        res = (
            self.StepScanToUsedQuadrupolePosition.isin(
                hpr_idxs_touching_the_quadrupole_position
            )
            .astype(int)
            .to_numpy()
        )
        res[res == 1] = 2
        res[self.StepScanToUsedQuadrupolePosition.isin(min_steps_quads).to_numpy()] = 3
        res[self.StepScanToUsedQuadrupolePosition.isin(max_steps_quads).to_numpy()] = 1
        return res

    @functools.cache
    def diagonal_scan_to_step(self, hpr_idx):
        subframe_matrix = self.get_subframe_matrix(hpr_idx)
        diagonal_scan_to_step = np.zeros(
            shape=(self.diagonal_cnt + 1, self.scan_cnt + 1), dtype=np.uint32
        )
        for scan in range(self.min_scan, self.max_scan + 1):
            for step in range(self.min_step, self.max_step + 1):
                diagonal = subframe_matrix[scan - 1, step]
                if diagonal != 0:
                    diagonal_scan_to_step[diagonal, scan] = step
        return diagonal_scan_to_step

    def plot_scans_and_steps_used_by_hypothetical_precursor_range(
        self, hpr_idx: int, suptitle_fontsize: int = 12, show: bool = True, **kwargs
    ) -> None:
        """Plot a matrix view of pairs (scan, step) that touch a given hypothetical precursor range.

        Arguments:
            hpr_idx (int): The number of the hypothetical precursor range.
            suptitle_fontsize (int): Suptitle font size.
            show (bool): Show the plot.
            **kwargs: Other parameters to the ax.matshow function used underneath.
        """
        import matplotlib.pyplot as plt

        assert (
            hpr_idx in self.HPR_intervals.index
        ), f"The range number you provided, hpr_idx={hpr_idx}, is outside the available range, i.e. between {self.HPR_intervals.index.min()} and {self.HPR_intervals.index.max()}."
        start, stop = self.HPR_intervals.loc[hpr_idx]
        subframe_df = self.get_subframe_matrix(hpr_idx)
        plt.matshow(subframe_df, aspect="auto", origin="lower", **kwargs)
        plt.yticks(ticks=self.scans - 1, labels=self.scans)
        plt.suptitle(
            f"HPR [{start}, {stop}], Minimal Coverage = {100*self.min_coverage_fraction}%",
            fontsize=suptitle_fontsize,
        )
        plt.xlabel("Step, i.e. MIDIA diagonal")
        plt.ylabel("Scan Number")
        plt.show()
        if show:
            plt.show()

    def _parse_input_hpr_indices(
        self,
        hpr_indices: list[int] | int | typing.Iterable[int] | None = None,
    ) -> set[int]:
        if hpr_indices is None:
            hpr_indices = list(self.HPR_intervals.index)
        elif isinstance(hpr_indices, int):
            hpr_indices = [hpr_indices]
        elif isinstance(hpr_indices, list):
            pass
        else:
            hpr_indices = list(hpr_indices)
        hpr_indices = set(hpr_indices)  # making sure it is unique
        # assert all are present:
        for hpr_idx in hpr_indices:
            assert (
                hpr_idx in self.HPR_intervals.index
            ), f"hpr_idx = {hpr_idx} is not among HPR_intervals"
        return hpr_indices

    def _get_hpr_data(
        self,
        hpr_idx: int,
        step: int,
        frame_data: MEASUREMENTS_TYPE,
    ) -> MEASUREMENTS_TYPE:
        try:
            min_scan, max_scan = self.hpr_step_to_scan_min_max[(hpr_idx, step)]
            if min_scan < max_scan:
                min_idx, max_idx = binary_search(
                    frame_data["scan"], min_scan, max_scan + 1
                )
                if min_idx < max_idx:
                    return {
                        col: measurements[min_idx:max_idx]
                        for col, measurements in frame_data.items()
                    }
        except KeyError:  # some hprs are simply not present in a given step.
            pass
        return {col: self.empty_data[col] for col in frame_data}

    def get_hpr_cycle_step_data(
        self,
        hpr_idx: int,
        cycle: int,
        step: int,
        columns: tuple[str] = ("scan", "tof", "intensity"),
    ) -> MEASUREMENTS_TYPE:
        frame = self.dia_run.cycle_step_to_ms2_frame(cycle, step)
        frame_data = self.dia_run.opentims.query(frame, columns=columns)
        return self._get_hpr_data(hpr_idx, step, frame_data)

    def iter_hprs(
        self,
        hpr_indices: list[int] | int | typing.Iterable[int] | None = None,
        progressbar: bool = False,
        columns=("scan", "tof", "intensity"),
    ) -> typing.Iterator[tuple[int, int, int, MEASUREMENTS_TYPE]]:
        """Iterate over hpr datasets.

        Arguments:
            hpr_indices (list,int,Iterable[int]): The indices of the HPRs to report.
            progressbar (bool): Show progressbar?

        Yields:
            tuple: the index of the current hpr, the cycle, the step, and the data consisting of a dictionary of numpy arrays with scans, tofs, and intensities.
        """
        hpr_indices = self._parse_input_hpr_indices(hpr_indices)
        MS2_frames_meta = self.dia_run.DiaFrameMsMsInfo[["cycle", "step"]].itertuples(
            index=False
        )
        if progressbar:
            MS2_frames_meta = tqdm(
                MS2_frames_meta, total=len(self.dia_run.DiaFrameMsMsInfo)
            )
        for cycle, step in MS2_frames_meta:
            frame = self.dia_run.cycle_step_to_ms2_frame(cycle, step)
            # one data query for all hprs: RAM likes it, Matteo likes it, Michał is furious
            frame_data = self.dia_run.opentims.query(frame, columns=columns)
            for hpr_idx in hpr_indices:
                data = self._get_hpr_data(hpr_idx, step, frame_data)
                yield (hpr_idx, cycle, step, data)

    # TODO: add columns support
    # TODO: replace collections.defaultdict(list) with a simpler list of empty arrays.
    def iter_cycle_aggregate_hprs(
        self,
        hpr_indices: list[int] | int | typing.Iterable[int] | None = None,
        progressbar: bool = False,
    ) -> typing.Iterator[tuple[int, int, MEASUREMENTS_TYPE]]:
        """Iterate over hypothetical precursor ranges aggregated over steps in one cycle.

        Arguments:
            hpr_indices (list,int,Iterable[int]): The indices of the HPRs to report.
            progressbar (bool): Show progressbar?

        Yields:
            tuple: the index of the current hpr, the cycle, and the data consisting of a dictionary of numpy arrays with scans, tofs, and intensities.
            The step information is aggregated out: intensities corresponding to the same tuples (scan,tof) are summed up.
        """
        _aggregate = lambda hpr_step_datasets: dict(
            zip(
                ("scan", "tof", "intensity"),
                dedup_v2(*combine_hpr_step_datasets(hpr_step_datasets)),
            )
        )
        hpr_indices = self._parse_input_hpr_indices(hpr_indices)
        hpr_idx_to_step_datasets = collections.defaultdict(list)
        #                                          this is the number of steps
        hprs_times_steps_count = len(hpr_indices) * (self.dia_run.max_step + 1)
        for i, (hpr_idx, cycle, step, data) in enumerate(
            self.iter_hprs(hpr_indices, progressbar)
        ):
            hpr_idx_to_step_datasets[hpr_idx].append(data)
            if (
                i + 1
            ) % hprs_times_steps_count == 0:  # +1 because we really want to count from one
                for hpr_idx, hpr_step_datasets in hpr_idx_to_step_datasets.items():
                    agg_data = _aggregate(hpr_step_datasets)
                    yield (hpr_idx, cycle, agg_data)
                hpr_idx_to_step_datasets = collections.defaultdict(list)

    def iter_hpr_transform(
        self,
        hpr_indices: list[int] | int | typing.Iterable[int] | None = None,
        columns: tuple[str] = ("scan", "tof", "intensity"),
        progressbar: bool = False,
    ) -> typing.Iterator[tuple[int, int, int, MEASUREMENTS_TYPE]]:
        steps_cnt = self.dia_run.max_step + 1
        hpr_indices = self._parse_input_hpr_indices(hpr_indices)
        hprs_times_steps_count = len(hpr_indices) * steps_cnt
        storage_templ = {
            (hpr_idx, diagonal, col): [np.empty(shape=(0,), dtype=np.uint32)]
            * steps_cnt
            for hpr_idx in hpr_indices
            for diagonal in self.diagonals
            for col in columns
        }  # the empty arrays are important: some of them remain there and concatenate needs that
        # also: we assured above that we have all of the diagonals present and in the right order.
        hpr_idx_diagonal_column_to_measurement = (
            storage_templ.copy()
        )  # massive speed-up over repeated construction...
        prev_cycle = -cmath.inf
        for i, (hpr_idx, cycle, step, data) in enumerate(
            self.iter_hprs(hpr_indices, progressbar)
        ):
            scan_step_to_group = self.get_subframe_matrix(hpr_idx)
            groups = scan_step_to_group[data["scan"] - 1, step]
            for diagonal, group_start, group_end in get_group_tags_starts_ends(groups):
                assert (
                    diagonal in self.diagonals
                ), f"Found a point that does not belong to HPR={hpr_idx}"
                for col, values in data.items():
                    hpr_idx_diagonal_column_to_measurement[hpr_idx, diagonal, col][
                        step
                    ] = values[group_start:group_end]
            if (i + 1) % hprs_times_steps_count == 0:
                assert prev_cycle < cycle, "Something went wrong with cycles."
                prev_cycle = cycle
                for hpr_idx in hpr_indices:
                    for diagonal in self.diagonals:
                        data = {
                            col: np.concatenate(
                                hpr_idx_diagonal_column_to_measurement[
                                    (hpr_idx, diagonal, col)
                                ]
                            )
                            for col in columns
                        }
                        yield hpr_idx, cycle, diagonal, data
                hpr_idx_diagonal_column_to_measurement = storage_templ.copy()


def get_max_chars_needed(xx: typing.Iterable[float]) -> int:
    return max(len(str(x)) for x in xx)


# TODO: add option to immediately translate hpr results into hdf.
# from enum import Enum
# class HPR_version(Enum):
#     SIMPLE = 1
#     CYCLE_STEP_AGGREGATE = 2
#     HPR_TRANSFORM = 3


def write_hprs(
    HPR_intervals: pd.DataFrame,
    source: pathlib.Path,
    target: pathlib.Path | None = None,
    hpr_type: str = "transform",
    min_coverage_fraction: float = 0.5,  # no unit
    window_width: float = 36.0,  # Da
    min_scan: int = 1,
    max_scan: int | None = None,
    compression_level: int = 1,
    make_all_frames_seem_unfragmented: bool = True,
    right_quadrupole_buffer: float = 0.001,
    verbose: bool = False,
    _max_iterations: int | None = None,
    _soft_limit: int = 4096,
) -> list[pathlib.Path]:
    """
    Arguments:
        HPR_intervals (pd.DataFrame): A data frame with columns 'hpr_start' and 'hpr_stop' describing the beginning and the end of intervals.
        source (pathlib.Path): Path to source .d folder.
        target (pathlib.Path): Path to target folder that will be filled with .d subfolders with individual HPR tdfs.
        combine_steps_per_cycle (bool): Should the steps be combined within a cycles for each individual HPR? It seems to work better with 4DFF.

    """
    # padding_for_floats = max(
    #     get_max_chars_needed(HPR_intervals.hpr_start),
    #     get_max_chars_needed(HPR_intervals.hpr_stop),
    # ) + 1
    # padding_for_ints = get_max_chars_needed(HPR_intervals.index) + 1
    padding_for_floats = 6
    padding_for_ints = 4
    num2str = lambda number, padding: str(number).zfill(padding).replace(".", "_")

    hpr_type_to_names = {
        "simple": ("HPRs", "HPR"),
        "transform": ("TransformHPRs", "TransformHPR"),
        "aggregate": ("AggregateHPRs", "AggHPR"),
    }
    try:
        (
            target_folder_name,
            subfolder_name,
        ) = hpr_type_to_names[hpr_type]
    except KeyError:
        raise KeyError(
            f"Unsupported type of HPRs: {hpr_type}. Supported: {', '.join(hpr_type_to_names)}."
        )

    if target is None:
        target = source / target_folder_name

    result_folders = [
        target
        / f"{subfolder_name}_{num2str(hpr.Index, padding_for_ints)}__{num2str(hpr.hpr_start, padding_for_floats)}__{num2str(hpr.hpr_stop, padding_for_floats)}.d"
        for hpr in HPR_intervals.itertuples()
    ]
    try:
        print(target)
        target.mkdir()
    except FileExistsError:
        for target_path in result_folders:
            assert_minimal_input_for_clusterings_exist(target_path)
        if verbose:
            print(f"Results were already there: not repeating.")
        return result_folders

    dia_run = DiaRun(
        fromwhat=source,
        preload_data=False,
        columns=("frame", "scan", "tof", "intensity"),
    )

    hprs = HPRS(
        HPR_intervals=HPR_intervals,
        dia_run=dia_run,
        min_coverage_fraction=min_coverage_fraction,
        window_width=window_width,
        right_quadrupole_buffer=right_quadrupole_buffer,
    )
    # highening limits on the number of simultaneously open file handlers on Linux
    soft_limit, hard_limit = get_limits()
    if soft_limit < _soft_limit:
        set_soft_limit(_soft_limit)

    if verbose:
        print(f"Copying analysis.tdf into {len(result_folders)} folders.")
    saviours = {
        hpr_index: SaveTIMS(
            opentims_obj=hprs.dia_run.opentims,
            path=outcome_folder,
            compression_level=compression_level,
        )
        for hpr_index, outcome_folder in zip(hprs.HPR_intervals.index, result_folders)
    }

    frame_to_NumScans = dict(zip(dia_run.Frames.Id, dia_run.Frames.NumScans))
    # TODO: in absence of only some hprs, might only create those missing...

    if hpr_type == "simple":
        for hpr_idx, cycle, data in itertools.islice(
            hprs.iter_cycle_aggregate_hprs(progressbar=verbose),
            _max_iterations,
        ):
            ms1_frame_in_the_given_cycle = hprs.dia_run.cycle_to_ms1_frame(cycle)
            saviour = saviours[hpr_idx]
            saviour.save_frame_tofs(
                scans=data["scan"],
                tofs=data["tof"],
                intensities=data["intensity"],
                total_scans=frame_to_NumScans[ms1_frame_in_the_given_cycle],
                src_frame=ms1_frame_in_the_given_cycle,
                run_deduplication=False,
                MsMsType=0,
            )
    elif hpr_type == "aggregate":
        # something is wrong here!
        cycle_step_to_NumScans = dict(
            zip(
                zip(
                    *hprs.dia_run.ms2_frame_to_cycle_step(
                        hprs.dia_run.DiaFrameMsMsInfo.Frame.to_numpy()
                    )
                ),
            )
        )
        for hpr_idx, cycle, step, data in itertools.islice(
            hprs.iter_hprs(progressbar=verbose),
            _max_iterations,
        ):
            saviour = saviours[hpr_idx]
            saviour.save_frame_tofs(
                scans=data["scan"],
                tofs=data["tof"],
                intensities=data["intensity"],
                total_scans=cycle_step_to_NumScans[(cycle, step)],
                src_frame=hprs.dia_run.cycle_step_to_ms2_frame(cycle, step),
                run_deduplication=False,
                MsMsType=0,
            )
    elif hpr_type == "transform":
        all_frames = hprs.dia_run.opentims.frames["Id"]
        all_retention_times = hprs.dia_run.opentims.frames["Time"]
        frame_to_NumScans = dict(
            zip(all_frames, hprs.dia_run.opentims.frames["NumScans"])
        )
        MS1_frames = all_frames[hprs.dia_run.opentims.frames["MsMsType"] == 0]
        # The idea here is to linearly interpolate within supplied values.
        diagonal_cycle_to_retention_time_frame_NumScans = {}
        for diagonal in hprs.diagonals:
            mock_frames = MS1_frames + 21 * (diagonal - 1) / len(hprs.diagonals)
            retention_times = np.interp(
                x=mock_frames,
                xp=all_frames,
                fp=all_retention_times,
                left=-1,  # blind guardian
                right=-1,  # blind guardian
            )
            frames = mock_frames.astype(int)
            frames = frames[retention_times > -1]
            retention_times = retention_times[retention_times > -1]
            cycles = hprs.dia_run.frame_to_cycle(frames)
            for cycle, rt, frame in zip(cycles, retention_times, frames):
                diagonal_cycle_to_retention_time_frame_NumScans[(diagonal, cycle)] = (
                    rt,
                    frame,
                    frame_to_NumScans[frame],
                )

        for hpr_idx, cycle, diagonal, data in itertools.islice(
            hprs.iter_hpr_transform(progressbar=verbose),
            _max_iterations,
        ):
            saviour = saviours[hpr_idx]
            (
                retention_time,
                frame,
                num_scans,
            ) = diagonal_cycle_to_retention_time_frame_NumScans[(diagonal, cycle)]
            saviour.save_frame_tofs(
                scans=data["scan"],
                tofs=data["tof"],
                intensities=data["intensity"],
                total_scans=num_scans,
                src_frame=frame,
                run_deduplication=False,
                MsMsType=0,
                Time=retention_time,
            )
    else:
        raise NotImplementedError(
            f"Algorithm for hpr_type={hpr_type} is not supported."
        )

    hprs.HPR_intervals.to_csv(path_or_buf=target / "HPR_intervals.csv")
    if verbose:
        print(
            "Updating the analysis.tdf files: in particular, writing down retention times of the MS1 frames within a given cycle."
        )
    # _Frames is the original frame info used for updates to the target analysis.tdf files
    # Here we read the original _Frames once only in under a second and avoid
    # reading them 200 times which takes 42"
    with sqlite3.connect(source / "analysis.tdf") as sqlite_connection:
        _Frames = pd.read_sql("SELECT * FROM Frames", sqlite_connection, index_col="Id")
    for saviour in saviours.values():
        saviour.close(_Frames=_Frames)
    return result_folders


def cli():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Prepare Hypothetical Precursor Ranges."
    )
    parser.add_argument(
        "source", metavar="<source.d>", help="source path", type=pathlib.Path
    )
    parser.add_argument(
        "target", metavar="<destination.d>", help="destination path", type=pathlib.Path
    )
    parser.add_argument(
        "--min_hpr_mz",
        help="The minimal m/z ratio in the sequence of HPRs.",
        type=float,
        default=300,
    )
    parser.add_argument(
        "--max_hpr_mz",
        help="The maximal m/z ratio in the sequence of HPRs.",
        type=float,
        default=1_500,
    )
    parser.add_argument(
        "--width_hpr_mz", help="The width in m/z of the HPRs.", type=float, default=12.0
    )
    parser.add_argument(
        "--min_coverage_fraction",
        help="The minimal coverage of the HPR by the quadrupole: provide a float fraction between 0 and 1.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--window_width",
        help="The presumed width of the quadrupole window.",
        default=36.0,
    )
    parser.add_argument(
        "--min_scan", help="The minimal scan to consider.", default=1, type=int
    )
    parser.add_argument(
        "--max_scan",
        help="The maximal scan to consider. By default, will impute the value.",
        default=None,
    )
    parser.add_argument(
        "--compression_level", help="Compression level used.", default=1, type=int
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print more info to stdout."
    )
    args = parser.parse_args()

    assert args.min_coverage_fraction >= 0, "min_coverage_fraction must be nonnegative"
    assert (
        args.min_coverage_fraction <= 1
    ), "min_coverage_fraction must smaller equal to 1"

    HPR_intervals = make_overlapping_HPR_mz_intervals(
        min_mz=args.min_hpr_mz,
        max_mz=args.max_hpr_mz,
        width=args.width_hpr_mz,
    )
    target_paths = write_hprs(
        HPR_intervals=HPR_intervals,
        source=args.source,
        target=args.target,
        min_coverage_fraction=args.min_coverage_fraction,  # no unit
        window_width=args.window_width,  # Da
        min_scan=args.min_scan,
        max_scan=args.max_scan,
        compression_level=args.compression_level,
        make_all_frames_seem_unfragmented=not args.leave_original_meta,
        verbose=args.verbose,
    )
    if args.verbose:
        print(f"Outcome .d folders:\n" + "\n".join(str(tp) for tp in target_paths))


if __name__ == "__main__":
    cli()
