from __future__ import annotations
from collections import namedtuple
from tqdm import tqdm

import itertools
import numpy as np
import pandas as pd
import pathlib
import typing


from savetimspy.common_assertions import (
    assert_minimal_input_for_clusterings_exist,
)
from savetimspy.fs_ops import reset_max_open_soft_file_handles
from savetimspy.interval_ops import NestedContainmentList
from savetimspy.numba_helper import coordinatewise_range
from savetimspy.pandas_ops import is_sorted
from dia_common.dia_main import DiaRun
from savetimspy.writer import SaveTIMS
from savetimspy.write_frame_datasets import FrameDataset



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
        min_scan: int|None = 1,
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

    def iter_hprs_per_cycle_step(self) -> typing.Iterator[HPRbyte]:
        """Iterate the tuples of cycle, step, hpr numbers, and the resulting hypothetical precursor range data.
        """
        cols_to_choose = ("scan","tof","intensity")
        tof_intensity = ["tof","intensity"]
        cycle_step_tuples = zip(
            self.dia_run.DiaFrameMsMsInfo.cycle, 
            self.dia_run.DiaFrameMsMsInfo.step,
        )
        for cycle, step in cycle_step_tuples:
            raw_peaks = self.dia_run.get_ms2_raw_peaks_dict(
                cycle=cycle, 
                step=step,
                columns=cols_to_choose,
            )
            raw_peaks = pd.DataFrame(raw_peaks) 
            raw_peaks = raw_peaks.set_index("scan")
            raw_peaks_prod = pd.merge(
                self.step_to_scan_hpr_dfs[step],
                raw_peaks,
                left_index=True,
                right_index=True,
            )
            for hpr_idx, data in raw_peaks_prod.groupby("hpr_idx")[tof_intensity]:
                yield HPRbyte(
                    cycle=cycle,
                    step=step,
                    hpr_idx=hpr_idx,
                    scans=data.index.to_numpy().astype(np.uint32),
                    tofs=data.tof.to_numpy(),
                    intensities=data.intensity.to_numpy(),
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
        hpr_bytes = iter(self.iter_hprs_per_cycle_step)
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

    def iter_nonempty_aggregated_cycle_hpr_data(self) -> typing.Iterator[ tuple[int,int,FrameDataset] ]:
        """
        Iterate over nonempty hypothetical precursor ranges datasets aggregate within each cycle.
        
        And so it come to being: the most complicated of all of the data preparation procedures...

        Yields:
            tuple: The cycle number, the HPR index (in the self.HPR_intervals table), and the FrameDataset used with write_frame_datasets file dumping procedure. 
        """
        # table reused in calculations
        step_scan_to_hpr_idx = [ ]
        for step, df in enumerate(self.step_to_scan_hpr_dfs):
            df = df.reset_index()
            df["step"] = step
            step_scan_to_hpr_idx.append(df)
        step_scan_to_hpr_idx = pd.concat(step_scan_to_hpr_idx, ignore_index=True)
        step_scan_to_hpr_idx.scan = step_scan_to_hpr_idx.scan.astype("uint32")
        step_scan_to_hpr_idx.step = step_scan_to_hpr_idx.step.astype("uint32")

        # deduplicate a dataframe with columns scan, tof and intensity, and sort by scan and tof
        dedup = lambda df: df.groupby(["scan","tof"], sort=True, as_index=False).intensity.sum()

        # The maximal NumScan per cycle across all the steps
        #   will be used as total scans, for a lack of better candidate.
        cycle2maxNumScans = self.dia_run.DiaFrameMsMsInfo.merge(
            self.dia_run.Frames[["Id","NumScans"]],
            left_on="Frame",
            right_on="Id"
        ).groupby("cycle").NumScans.max().to_numpy()


        for cycle in range(self.dia_run.min_cycle, self.dia_run.max_cycle+1):
            # getting indices:
            steps_per_cycle = np.arange(self.dia_run.min_step, self.dia_run.max_step+1)
            ms2_frame_ids_per_cycle = self.dia_run.cycle_step_to_ms2_frame(cycle,steps_per_cycle)
            ms1_frame_id_in_this_cycle = self.dia_run.cycle_to_ms1_frame(cycle)

            # getting raw data
            raw_peaks_per_cycle = self.dia_run._get_frames_raw( frame_ids=ms2_frame_ids_per_cycle,
                                                                columns=("frame","scan","tof","intensity") )
            raw_peaks_per_cycle['step'] = self.dia_run.ms2_frame_to_step(raw_peaks_per_cycle["frame"])
            del raw_peaks_per_cycle['frame']# not needed, steps in a given cycle encode it
            raw_peaks_per_cycle = pd.DataFrame(raw_peaks_per_cycle)
            raw_hprs_peaks = pd.merge_ordered(# much faster than pd.merge
                step_scan_to_hpr_idx,
                raw_peaks_per_cycle,
                on = ["step","scan"],
                how="inner"
            )
            #NOTE: here we OMIT STEPS: but we might not do it and save it?    we can put it somewher here   ⬇
            for hpr_idx, hpr_data_per_cycle in raw_hprs_peaks.groupby("hpr_idx")[ ["scan","tof","intensity", ] ]:
                hpr_data_per_cycle = dedup( hpr_data_per_cycle )
                framedataset = FrameDataset(
                    total_scans = cycle2maxNumScans[cycle],
                    src_frame = ms1_frame_id_in_this_cycle,
                    df=hpr_data_per_cycle,
                )
                yield (cycle, hpr_idx, framedataset)

    def iter_all_aggregated_cycle_hpr_data(self, verbose: bool=False) -> typing[tuple[int,int,FrameDataset]]:
        """
        Iterate over all hypothetical precursor ranges datasets aggregate within each cycle.

        
        Yields:
            tuple: The cycle number, the HPR index (in the self.HPR_intervals table), and the FrameDataset used with write_frame_datasets file dumping procedure. 
        """
        cycle_hpr_idx_framedataset_tuples = iter(self.iter_nonempty_aggregated_cycle_hpr_data())
        empty_df = pd.DataFrame(columns=("scan","tof","intensity"))
        cycle2maxNumScans = self.dia_run.DiaFrameMsMsInfo.merge(
            self.dia_run.Frames[["Id","NumScans"]],
            left_on="Frame",
            right_on="Id"
        ).groupby("cycle").NumScans.max().to_numpy()

        cycles = np.arange(self.dia_run.min_cycle, self.dia_run.max_cycle+1)
        hpr_indices = self.hpr_indices
        cycle_hpr_idx_tuples = itertools.product(cycles, hpr_indices)
        if verbose:
            cycle_hpr_idx_tuples = tqdm(cycle_hpr_idx_tuples, total=len(cycles)*len(hpr_indices))

        # guards
        prev_cycle = -1
        prev_hpr_idx = -1
        try:
            prev_cycle, prev_hpr_idx, prev_framedataset = next(cycle_hpr_idx_framedataset_tuples)
        except StopIteration:
            pass# cannot stop because emtpying this sequence: need to continue supply of empty data

        # we get the main sequence of events
        # the loop below is trying to catch up with it
        # if it catches up, it reports the element from the origal sequence
        # otherwise, it report empty data.
        for cycle, hpr_idx in cycle_hpr_idx_tuples:
            if cycle == prev_cycle and hpr_idx == prev_hpr_idx:
                yield (prev_cycle, prev_hpr_idx, prev_framedataset)
                try:
                    prev_cycle, prev_hpr_idx, prev_framedataset = next(cycle_hpr_idx_framedataset_tuples)
                except StopIteration:
                    pass
            else:
                framedataset = FrameDataset(total_scans = cycle2maxNumScans[cycle],
                                            src_frame = self.dia_run.cycle_to_ms1_frame(cycle),
                                            df = empty_df )
                yield (cycle, hpr_idx, framedataset)



def get_max_chars_needed(xx: typing.Iterable[float]) -> int:
    return max( len(str(x)) for x in xx )



def write_hprs(
    HPR_intervals: pd.DataFrame,
    source: pathlib.Path,
    target: pathlib.Path|None = None,
    combine_steps_per_cycle: bool=True,
    min_coverage_fraction: float=0.5,# no unit
    window_width: float=36.0,# Da
    min_scan: int=1,
    max_scan: int|None = None,
    compression_level: int=1,
    make_all_frames_seem_unfragmented: bool=True,
    verbose: bool=False,
) -> list[pathlib.Path]:
    padding_for_floats = max(
        get_max_chars_needed(HPR_intervals.hpr_start),
        get_max_chars_needed(HPR_intervals.hpr_stop),
    ) + 1
    padding_for_ints = get_max_chars_needed(HPR_intervals.index) + 1
    num2str = lambda number, padding: str(number).zfill(padding).replace(".","_")
    if target is None:
        target = source/"HPRs"
    result_folders = [
        target/f"HPR_{num2str(hpr.Index, padding_for_ints)}__{num2str(hpr.hpr_start, padding_for_floats)}__{num2str(hpr.hpr_stop, padding_for_floats)}.d"
        for hpr in HPR_intervals.itertuples()
    ]
    try:
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
        HPR_intervals = HPR_intervals,
        dia_run = dia_run,
        min_coverage_fraction = min_coverage_fraction,
        window_width = window_width,
        verbose=verbose,
    )

    reset_max_open_soft_file_handles(verbose=verbose)
    saviours = {
        hpr_index: SaveTIMS(
            opentims_obj=hprs.dia_run.opentims,
            path=outcome_folder,
            compression_level=compression_level,
        ) for hpr_index, outcome_folder in zip(hprs.HPR_intervals.index, result_folders)
    }

    if combine_steps_per_cycle:
        for cycle, hpr_idx, frame_dataset in hprs.iter_all_aggregated_cycle_hpr_data(verbose=verbose):
            saviour = saviours[hpr_idx]
            saviour.save_frame_tofs(
                scans=frame_dataset.df.scan.to_numpy(),
                tofs=frame_dataset.df.tof.to_numpy(),
                intensities=frame_dataset.df.intensity.to_numpy(),
                total_scans=frame_dataset.total_scans,
                src_frame=frame_dataset.src_frame,
                run_deduplication=False,
                set_MsMsType_to_0=True,
            )
    else:
        MS2Frames = pd.DataFrame(hprs.dia_run.opentims.frames).query("MsMsType > 0")
        cycle_step_to_NumScans = dict(zip(
            zip(*hprs.dia_run.ms2_frame_to_cycle_step(MS2Frames.Id)),
            hprs.dia_run.opentims.frames["NumScans"]
        ))
        for cycle, step, hpr_idx, scans, tofs, intensities in hprs.full_iter(verbose=True):
            saviour = saviours[hpr_idx]
            saviour.save_frame_tofs(
                scans=scans,
                tofs=tofs,
                intensities=intensities,
                total_scans=cycle_step_to_NumScans[(cycle, step)],
                copy_sql=True,
                run_deduplication=False,
                set_MsMsType_to_0=True,
                src_frame=dia_run.cycle_to_ms1_frame(cycle),
            )

    del saviours
    
    hprs.HPR_intervals.to_csv(path_or_buf=target/"HPR_intervals.csv")
    
    return result_folders






if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='Prepare Hypothetical Precursor Ranges.')
    parser.add_argument("source", metavar="<source.d>", help="source path", type=pathlib.Path)
    parser.add_argument("target", metavar="<destination.d>", help="destination path", type=pathlib.Path)
    parser.add_argument("target", metavar="<destination.d>", help="destination path", type=pathlib.Path)
    parser.add_argument("--min_hpr_mz", help="The minimal m/z ratio in the sequence of HPRs.", type=float, default=300)
    parser.add_argument("--max_hpr_mz", help="The maximal m/z ratio in the sequence of HPRs.", type=float, default=1_500)
    parser.add_argument("--width_hpr_mz", help="The width in m/z of the HPRs.", type=float, default=12.0)
    parser.add_argument("--min_coverage_fraction", help="The minimal coverage of the HPR by the quadrupole: provide a float fraction between 0 and 1.", type=float, default=0.5)
    parser.add_argument("--window_width", help="The presumed width of the quadrupole window.", default=36.0)
    parser.add_argument("--min_scan", help="The minimal scan to consider.", default=1, type=int)
    parser.add_argument("--max_scan", help="The maximal scan to consider. By default, will impute the value.", default=None)
    parser.add_argument("--compression_level", help="Compression level used.", default=1, type=int)
    parser.add_argument("--verbose", 
        action='store_true',
        help='Print more info to stdout.')
    args = parser.parse_args()

    assert args.min_coverage_fraction >= 0, "min_coverage_fraction must be nonnegative"
    assert args.min_coverage_fraction <= 1, "min_coverage_fraction must smaller equal to 1"

    HPR_intervals = make_overlapping_HPR_mz_intervals(
        min_mz = args.min_hpr_mz,
        max_mz = args.max_hpr_mz,
        width = args.width_hpr_mz,
    )
    target_paths = write_hprs(
        HPR_intervals=HPR_intervals,
        source=args.source,
        target=args.target,
        min_coverage_fraction=args.min_coverage_fraction,# no unit
        window_width=args.window_width,# Da
        min_scan=args.min_scan,
        max_scan=args.max_scan,
        compression_level=args.compression_level,
        make_all_frames_seem_unfragmented=not args.leave_original_meta,
        verbose=args.verbose,
    )
    if args.verbose:
        print(f"Outcome .d folders:\n"+"\n".join(str(tp) for tp in target_paths))
