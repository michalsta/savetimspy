import io
import numpy as np
import pandas as pd
import zstd


def get_data_as_bytes(
    scans: np.array,
    tofs: np.array,
    intensities: np.array,
    total_scans: int
) -> bytearray:
    """Save one frame a bytearray."""
    # preparing peak counts per each scan
    total_scans = int(total_scans)
    peak_cnts = [total_scans]
    ii = 0
    for scan_id in range(1, total_scans):
        counter = 0
        while ii < len(scans) and scans[ii] < scan_id:
            ii += 1
            counter += 1
        peak_cnts.append(counter*2)
    #peak_cnts[0] = len(scans)
    #peak_cnts.append(0)
    peak_cnts = np.array(peak_cnts, np.uint32)

    last_tof = -1
    last_scan = 0
    for ii in range(len(tofs)):
        if last_scan != scans[ii]:
            last_tof = -1
            last_scan = scans[ii]
        val = tofs[ii]
        tofs[ii] = val - last_tof
        last_tof = val

    assert total_scans >= last_scan, "strange! [Were you perhaps expecting a helpful message?! buahahaha!]"

    if not isinstance(intensities, np.ndarray) or not intensities.dtype == np.uint32:
        intensities = np.array(intensities, np.uint32)

    # copy-copy, you naugthy Startrek you
    interleaved = np.vstack([tofs, intensities]).transpose().reshape(len(scans)*2)
    back_data = peak_cnts.tobytes() + interleaved.tobytes()
    real_data = bytearray(len(back_data))

    reminder = 0
    bd_idx = 0
    for rd_idx in range(len(back_data)):
        if bd_idx >= len(back_data):
            reminder += 1
            bd_idx = reminder
        real_data[rd_idx] = back_data[bd_idx]
        bd_idx += 4

    return real_data


def compress_data(
    data: bytearray,
    compression_level:int=1
) -> bytearray:
    return zstd.ZSTD_compress(bytes(data), compression_level)


def write_frame_bytearray_to_open_file(
    file,
    data: bytearray,
    total_scans: int,
) -> None:
    file.write((len(data)+8).to_bytes(4, 'little', signed = False))
    file.write(int(total_scans).to_bytes(4, 'little', signed = False))
    file.write(data)


def dump_one_ready_frame_df_to_tdf(
    frame_df: pd.DataFrame,
    open_file_handler: io.BufferedWriter,
    total_scans: int,
) -> None:
    """ 
    Dump data with only one frame into tdf.

    Ready: containing columns "scan", "tof", and "intensity", no duplicate rows, and sorted in that order.
    """
    for column_name in ("scan","tof","intensity"):
        assert column_name in frame_df, f"Missing '{column_name}' column."
    total_scans = int(total_scans)
    frame_bytes = get_data_as_bytes(
        scans = frame_df.scan.values,
        tofs = frame_df.tof.values,
        intensities = frame_df.intensity.values,
        total_scans = total_scans)
    compressed_data = compress_data(
        data=frame_bytes,
        compression_level=1)
    write_frame_bytearray_to_open_file(
        file=open_file_handler,
        data=compressed_data,
        total_scans=total_scans)
