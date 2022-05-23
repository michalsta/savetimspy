from __future__ import annotations
from collections import namedtuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import pathlib
import typing


from savetimspy.numba_helper import coordinatewise_range
from savetimspy.interval_ops import NestedContainmentList
from savetimspy.pandas_ops import is_sorted



def make_overlapping_HPR_mz_intervals(
    min_mz: float=300.0,
    max_mz: float=1500.0,
    width:  float=12.0,
    add_overlaps: bool=True,
) -> pd.DataFrame:
    mz_borders = []
    mz_border = min_mz
    while mz_border <= max_mz:
        mz_borders.append(mz_border)
        mz_border += width 
    mz_borders = np.array(mz_borders)
    HPR_intervals = pd.DataFrame({
        "hpr_start": mz_borders[:-1],
        "hpr_stop":  mz_borders[1:],
    })
    if add_overlaps:
        HPR_intervals = pd.concat((HPR_intervals, (HPR_intervals + 6).iloc[:-1]))
        HPR_intervals = HPR_intervals.sort_values(["hpr_start","hpr_stop"])
        HPR_intervals.index = range(len(HPR_intervals))
    HPR_intervals.index.name = "hpr_idx"
    return HPR_intervals


def infer_the_max_usable_scan(DiaFrameMsMsWindows: pd.DataFrame) -> int:
    """Infer the maximal usable scan.

    After each fragmentation the quadrupole needs few last scans to be reset to the proper position.
    This does not seem to be the case for the last frame.
    Data collected during this resetting is weird and we neglect it.
    """
    return int(DiaFrameMsMsWindows.groupby("step").ScanNumBegin.max()[:-1].min())


HPRbyte = namedtuple("HPRbyte", "cycle step hpr_idx scans tofs intensities")

class HPRS:
    def __init__(
        self,
        HPR_intervals: pd.DataFrame,
        dia_run: DiaRun,
        min_coverage_fraction: float = 0.5,
        window_width: float|None = 36.0,
        min_scan: int|None = 0,
        max_scan: int|None = None,
        verbose: bool = False,
    ):
        self.HPR_intervals = HPR_intervals
        self.hpr_indices = HPR_intervals.index.to_numpy()
        self.dia_run = dia_run
        self.min_coverage_fraction = min_coverage_fraction
        self.window_width = window_width
        self.verbose = verbose

        # defining quadrupole positions
        self.DiaFrameMsMsWindows = self.dia_run.DiaFrameMsMsWindows
        self.steps = self.DiaFrameMsMsWindows.step.unique()
        if self.window_width is None:
            quad_borders_exp = \
            """
                quadrupole_start = IsolationMz - IsolationWidth / 2.0
                quadrupole_stop  = IsolationMz + IsolationWidth / 2.0
            """
        else:
            quad_borders_exp = \
            """
                quadrupole_start = IsolationMz - @self.window_width / 2.0
                quadrupole_stop  = IsolationMz + @self.window_width / 2.0
            """
        self.DiaFrameMsMsWindows = self.DiaFrameMsMsWindows.eval(quad_borders_exp)

        # defining scans of interest
        self.min_scan = min_scan
        if max_scan is None:
            self.max_scan = infer_the_max_usable_scan(self.DiaFrameMsMsWindows)

        self.scans = np.arange(self.min_scan, self.max_scan+1)
        # Mapping (step,scan) to actual quadrupole positions
        self.StepScanToQuadrupole = self.DiaFrameMsMsWindows.loc[
            np.repeat(
                self.DiaFrameMsMsWindows.index,
                self.DiaFrameMsMsWindows.eval("ScanNumEnd - ScanNumBegin")
            ),# repeats the index the total number of scan times
            [# result must contain:
                "step",
                "quadrupole_start",
                "quadrupole_stop",
                "CollisionEnergy",
                "IsolationMz",
            ]
        ].reset_index().rename(columns={"index":"quadrupole_idx"})
        self.StepScanToQuadrupole["scan"] = coordinatewise_range(
            # silly convention: scans start from 1
            self.DiaFrameMsMsWindows.ScanNumBegin.to_numpy() + 1,
            self.DiaFrameMsMsWindows.ScanNumEnd.to_numpy() + 1,
        )
        self.StepScanToQuadrupolePosition = self.StepScanToQuadrupole.pivot(
            index='scan',
            columns="step",
            values="quadrupole_idx",
        ).fillna(-1).astype(int)
        # convention: -1 above = no valid quadrupole position mapped

        scans_below_min_above_max = set(
            self.StepScanToQuadrupole.query(
                f"scan < {self.min_scan} or scan > {self.max_scan}"
            ).quadrupole_idx
        )
        self.StepScanToQuadrupolePosition[
            self.StepScanToQuadrupolePosition.isin(scans_below_min_above_max)
        ] = -1
        self.StepScanToQuadrupole = self.StepScanToQuadrupole[~self.StepScanToQuadrupole.scan.isin(scans_below_min_above_max)]


        # mapping hypothetical precursor ranges to quadrupole positions
        #   * initially every intersecting quadrupole position is valid
        #   * shortlisting quadrupole positions that 
        interval_db = NestedContainmentList.from_df(HPR_intervals)
        StepScanToQuadrupole_idxs, hpr_idxs = interval_db.query_df(
            self.StepScanToQuadrupole[["quadrupole_start", "quadrupole_stop"]]
        )

        self.hpr_quadrupole_matches = pd.concat([
            self.StepScanToQuadrupole.loc[StepScanToQuadrupole_idxs].reset_index(),
            self.HPR_intervals.loc[hpr_idxs].reset_index()
        ], axis=1)
        self.hpr_quadrupole_matches["quadrupole_coverage_dalton"] = \
            np.minimum(self.hpr_quadrupole_matches.hpr_stop, 
                       self.hpr_quadrupole_matches.quadrupole_stop) - \
            np.maximum(self.hpr_quadrupole_matches.hpr_start,
                       self.hpr_quadrupole_matches.quadrupole_start)
        
        self.hpr_quadrupole_matches["quadrupole_coverage_fraction"] = \
            self.hpr_quadrupole_matches.quadrupole_coverage_dalton / \
            (self.hpr_quadrupole_matches.hpr_stop - self.hpr_quadrupole_matches.hpr_start)

        self.hpr_quadrupole_matches = self.hpr_quadrupole_matches.query("quadrupole_coverage_fraction >= @self.min_coverage_fraction").copy()

        self.hpr_quadrupole_matches = self.hpr_quadrupole_matches.reset_index(drop=True)# don't need that old index

        # Setting some step, scan positions to unuseful:
        self.StepScanToUsedQuadrupolePosition = self.StepScanToQuadrupolePosition.copy()
        self.StepScanToUsedQuadrupolePosition[
            ~self.StepScanToUsedQuadrupolePosition.isin(
                self.hpr_quadrupole_matches.quadrupole_idx.unique()
            )
        ] = -1
        self.UnusedStepScanPositionsCount = sum(self.StepScanToUsedQuadrupolePosition.values == -1).sum()

        # Making a list of data frames that each map scan to hpr_idx
        # this is used in the iterator to map hpr_idx back unto the events.
        empty_df = pd.DataFrame(columns=["hpr_idx"])
        empty_df.index.name="scan"
        self.step_to_scan_hpr_dfs = [ empty_df ]*( self.dia_run.max_step+1 )# List must have all scan numbers
        for step, data in self.hpr_quadrupole_matches.groupby("step")[["scan","hpr_idx"]]:
            data["scan"] = data.scan.astype(np.uint32)
            self.step_to_scan_hpr_dfs[step] = data.set_index("scan")
        for step, step_to_scan_hpr_df in enumerate(self.step_to_scan_hpr_dfs):
            assert is_sorted(step_to_scan_hpr_df.index.values), f"Step's {step} step_to_scan_hpr_df scan index ain't sorted."

    def plot_scans_and_steps_used_by_hypothetical_precursor_range(
        self, 
        hpr_idx: int,
        fontsize: int=12,
        show: bool=True,
        **kwargs
    ) -> None:
        """Plot a matrix view of pairs (scan, step) that touch a given hypothetical precursor range.

        Arguments:
            hpr_idx (int): The number of the hypothetical precursor range.
            fontsize (int): Suptitle font size.
            show (bool): Show the plot.
            **kwargs: Other parameters to the ax.matshow function used underneath.
        """
        import matplotlib.pyplot as plt
        assert hpr_idx in self.HPR_intervals.index, f"The range number you provided, hpr_idx={hpr_idx}, is outside the available range, i.e. between {self.HPR_intervals.index.min()} and {self.HPR_intervals.index.max()}."
        start, stop = self.HPR_intervals.loc[hpr_idx]
        fig, ax = plt.subplots()
        ax.set_xticks(self.steps[:-1]+.5, minor=True)
        ax.xaxis.grid(True, which='minor', linewidth=.2)
        ax.set_yticks(self.scans[:-1]+.5, minor=True)
        ax.yaxis.grid(True, which='minor', linewidth=.2)
        hpr_idxs_touching_the_quadrupole_position = set(
            self.hpr_quadrupole_matches.query("hpr_idx == @hpr_idx").quadrupole_idx
        )
        ax.matshow(
            self.StepScanToUsedQuadrupolePosition.isin(
                hpr_idxs_touching_the_quadrupole_position
            ),
            aspect='auto',
            **kwargs
        )
        fig.suptitle(
            f"HPR [{start}, {stop}], Minimal Coverage = {100*self.min_coverage_fraction}%", 
            fontsize=fontsize
        )
        ax.set_xlabel("Step, i.e. MIDIA diagonal")
        ax.set_ylabel("Scan Number")
        if show:
            plt.show()

    def __iter__(self) -> typing.Iterator[tuple[int,int,int,np.array,np.array,np.array]]:
        """Iterate the tuples of cycle, step, hpr numbers, and the resulting hypothetical precursor range data.
        """
        cols_to_choose = ("scan","tof","intensity")
        tof_intensity = ["tof","intensity"]
        cycle_step_tuples = zip(
            self.dia_run.DiaFrameMsMsInfo.cycle, 
            self.dia_run.DiaFrameMsMsInfo.step,
        )
        # if self.verbose:
        #     cycle_step_tuples = tqdm(
        #         cycle_step_tuples,
        #         total=len(self.dia_run.DiaFrameMsMsInfo)
        #     )
        for cycle, step in cycle_step_tuples:
            raw_peaks = self.dia_run.get_ms2_raw_peaks_dict(
                cycle=cycle, 
                step=step,
                columns=cols_to_choose,
            )
            # The easiest thing I could think off...
            raw_peaks = pd.DataFrame(raw_peaks) 
            raw_peaks = raw_peaks.set_index("scan")
            raw_peaks_prod = pd.merge(
                self.step_to_scan_hpr_dfs[step],
                raw_peaks,
                left_index=True,
                right_index=True,
            )
            # yield raw_peaks_prod
            for hpr_idx, data in raw_peaks_prod.groupby("hpr_idx")[tof_intensity]:
                yield HPRbyte(
                    cycle,
                    step,
                    hpr_idx,
                    data.index.to_numpy().astype(np.uint32),
                    data.tof.to_numpy(),
                    data.intensity.to_numpy(),
                )

    def full_iter(self, verbose: bool=False) -> typing.Iterator[tuple[int,int,int,np.array,np.array,np.array]]:
        cycle_step_tuples = zip(
            self.dia_run.DiaFrameMsMsInfo.cycle, 
            self.dia_run.DiaFrameMsMsInfo.step,
        )
        if verbose:
            cycle_step_tuples = tqdm(
                cycle_step_tuples,
                total=len(self.dia_run.DiaFrameMsMsInfo)
            )
        empty_scan = np.array([], dtype=np.uint32)
        empty_tof = np.array([], dtype=np.uint32)
        empty_intensities = np.array([], dtype=np.uint32)
        hpr_bytes = iter(self)
        finished_hpr_bytes = False
        try:
            hpr_byte = next(hpr_bytes)
        except StopIteration:
            finished_hpr_bytes = True
        for cycle, step in cycle_step_tuples:
            for hpr_idx in self.hpr_indices:
                if not finished_hpr_bytes and hpr_byte.cycle == cycle and hpr_byte.step == step and hpr_byte.hpr_idx == hpr_idx:
                    yield hpr_byte
                    try:
                        hpr_byte = next(hpr_bytes)
                    except StopIteration:
                        finished_hpr_bytes = True
                else:
                    yield (cycle, step, hpr_idx, empty_scan, empty_tof, empty_intensities)


def write_hprs(
    source: pathlib.Path,
    target: pathlib.Path,
    frame_indices: typing.Union[npt.NDArray[np.uint32], int, typing.Iterable[int]],
    compression_level: int=1,
    make_all_frames_seem_unfragmented: bool=True,
    verbose: bool=False,
) -> pathlib.Path:
    pass



