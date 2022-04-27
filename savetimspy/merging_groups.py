import numpy as np
import opentimspy
import pathlib

from tqdm import tqdm
from typing import Tuple, Iterator, Iterable, Any, List

from savetimspy import SaveTIMS



def iter_consecutive_int_k_mers(
    max_int: int,
    k: int=3,
    min_int: int=0
) -> Iterator[Tuple[int]]:
    """Iterate tuples of consecutive integers, first integer always one higher.

    E.g. for min_int=3 max_int=n and k=3:
    (1,2,3),
    (2,3,4),
    (3,4,5),
    ...
    (n-2,n-1,n)

    Arguments:
        max_int (int): Maximal integer in series.
        k (int): Size of each tuple.
        min_int (int): Minimal integer in series.
    """
    for n in range(k+min_int, max_int+1):
        yield tuple(range(n-k, n))



def unequal_zip(
    *lst_of_iterables: Iterable[Any]
) -> Iterator[List[Any]]:
    """zip iterables of unequal sizes into tuples of potentially different length.

    Motivation: while zipping unequal lists the result is cut to the length of the shortest list.
    Here we do the oposite: we continue supplying tuples with different sizes until all lists are emptied.

    """
    iterables = {
        i: iter(it) for i, it in enumerate(lst_of_iterables)
    }
    while iterables:
        curr_zip = []
        for iterable_num, iterable in iterables.copy().items():
            try:
                curr_zip.append(next(iterable))
            except StopIteration:
                del iterables[iterable_num]
        if curr_zip:
            yield curr_zip
        else:
            break


def test_unequal_zip():
    lst_of_iterables = [
        range(10),
        range(9),
        range(5),
        range(35),
    ]
    print(list(unequal_zip(*lst_of_iterables)))
    


def iter_window_groups_k_mers_and_frames(
    rawdata: opentimspy.OpenTIMS,
    group_overlap: int=3
) -> Tuple[List[int], np.array]:
    """Iterate over window group tuples and their corresponding data.

    Arguments:
        rawdata (opentimspy.OpenTIMS)"""
    frame2windowgroup = rawdata.table2dict("DiaFrameMsMsInfo")
    window_groups = np.sort(np.unique(frame2windowgroup["WindowGroup"]))
    for group_indices in iter_consecutive_int_k_mers(
        min_int=min(window_groups),
        max_int=max(window_groups)+1,
        k=group_overlap,
    ):
        frames_groups = [
            frame2windowgroup["Frame"][frame2windowgroup["WindowGroup"] == gr] for gr in group_indices
        ] 
        frames_to_merge = list(unequal_zip(*frames_groups))
        yield (
            group_indices, 
            frames_to_merge,
        )



def save_moving_k_mer_MIDIA_diagonals(
    source_tdf_folder: str,
    destination_folder: str,
    group_overlap: int=3,
    verbose: bool=False,
) -> None:
    """Extract and merge MIDIA groups and save into consecutive frames in a moving window fashion.

    
    """
    rawdata = opentimspy.OpenTIMS(source_tdf_folder)
    frame2scans = {
        int(_id): int(_scan_cnt)
        for _id, _scan_cnt in zip(
            rawdata.frames["Id"],
            rawdata.frames["NumScans"]
        )
    }
    destination_folder = pathlib.Path(destination_folder)
    destination_folder.mkdir(parents=True)
    group_frames_to_merge = list(iter_window_groups_k_mers_and_frames(
        rawdata=rawdata,
        group_overlap=group_overlap,
    ))
    if verbose:
        print("Welcome!")
        print(f"We are going to extract MS2 MIDIA cycles in groups of {group_overlap}.")
        print("The following groups will be considered:")
        print([g for g,_ in group_frames_to_merge])
    for group_indices, frames_to_merge in group_frames_to_merge:
        destination_subfolder = destination_folder/("MIDIA_"+"_".join(str(g) for g in group_indices) + ".d")
        savetims = SaveTIMS(rawdata, destination_subfolder)
        if verbose:
            print(f"Saving windows to {destination_subfolder}")
            frames_to_merge = tqdm(list(frames_to_merge))
        for frames in frames_to_merge:
            D = rawdata.query(
                frames,
                columns='scan tof intensity'.split()
            )
            n_scans = max(frame2scans[gr] for gr in group_indices)
            savetims.save_frame_dict(D, n_scans)
