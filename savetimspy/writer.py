from __future__ import annotations
import os
import shutil
import opentimspy
from pathlib import Path
import cffi
import sqlite3
import numpy as np
import pandas as pd
import typing
import numpy.typing as npt
import zstd
import opentims_bruker_bridge
from savetimspy.numba_helper import (
    deduplicate,
    get_peak_cnts,
    modify_tofs,
    np_zip,
    get_realdata,    
)
from collections import namedtuple

so_path = opentims_bruker_bridge.get_appropriate_so_path()

ffi = cffi.FFI()
ffi.cdef('''

uint64_t tims_open (
    const char *analysis_directory_name,
    uint32_t use_recalibrated_state
    );

void tims_close (uint64_t handle);

uint32_t tims_mz_to_index(uint64_t handle,
     int64_t frame_id,
     const double *mz,
     double *index,
     uint32_t cnt
);

uint32_t tims_index_to_mz(uint64_t handle,
     int64_t frame_id,
     const double *mz,
     double *index,
     uint32_t cnt
);

uint32_t tims_get_last_error_string (char *buf, uint32_t len);
''')

ctims = ffi.dlopen(so_path)

def cast(type_str, np_arr):
    return ffi.cast(type_str, ffi.from_buffer(np_arr))

def get_err():
    s = ffi.new("char[10000]")
    ctims.tims_get_last_error_string(s, 10000)
    #return s.decode()
    return ''.join(x.decode() for x in s)


def update_frames_table(
    frame_row_updates: typing.Iterable[tuple[int, dict]],
    sqlite_connection: sqlite3.Connection,
    _cols: tuple=(
        'Id',
        'Time',
        'Polarity',
        'ScanMode',
        'MsMsType',
        'TimsId',
        'MaxIntensity',
        'SummedIntensities',
        'NumScans',
        'NumPeaks',
        'MzCalibration',
        'T1',
        'T2',
        'TimsCalibration',
        'PropertyGroup',
        'AccumulationTime',
        'RampTime',
    ),
    _Frames: pd.DataFrame|None=None,
) -> None:
    """Update the analysis.tdf::Frames table.

    Arguments:
        frame_row_updates (Iterable of tuples of int and dict): A sequence of tuples with the numbers of the Frame Ids from the sqlite to reuse and a dictionary with updates.
        sqlite_connection (sqlite3.Connection): An open connection to analysis.tdf.
        _cols (tuple): Columns in the Frames table.
        _Frames (pd.DataFrame): The contents of the original analysis.tdf:::Frames table. If not provided will be read in from the provided connection.
    """
    orig_rows, updates = zip(*frame_row_updates)
    orig_rows = np.array(orig_rows)
    updates = pd.DataFrame(updates)
    if _Frames is None:
        _Frames = pd.read_sql('SELECT * FROM Frames', sqlite_connection, index_col="Id")
    _Frames = _Frames.loc[orig_rows].reset_index()
    _Frames.update(updates)
    sqlite_connection.execute("DELETE FROM Frames")
    sqlite_connection.commit()# Not sure if necessary
    _Frames = _Frames[list(_cols)]
    sqlite_connection.executemany(
        f"INSERT INTO Frames({','.join(_cols)}) VALUES ({','.join('?'*len(_cols))})",
        _Frames.itertuples(index=False)
    )
    sqlite_connection.commit()# Not sure if necessary


# changes: 
# * after copy of the original sqlite no additional ops are done on it:
#   * rationale: I have no idea how sqlite works with multiple handles on it.
class SaveTIMS:
    def __init__(self, opentims_obj, path, compression_level = 1):
        self.src_tims_id = None
        self.tdf_bin = None
        self.sqlcon = None
        os.mkdir(path)
        self.opentims = opentims_obj
        self.src_tims_id = ctims.tims_open(bytes(str(opentims_obj.analysis_directory), 'ascii'), 0)
        self.dst_path = Path(path)
        self.db_path = self.dst_path / 'analysis.tdf'
        shutil.copyfile(opentims_obj.analysis_directory/'analysis.tdf', self.db_path)
        self.current_frame = 1# that's the Brukies' convention.
        self.tdf_bin = open(self.dst_path / 'analysis.tdf_bin', 'wb')
        self.sqlcon = sqlite3.connect(self.db_path) 
        self.compression_level = compression_level
        self.frame_row_updates = [] # will contain tuples (int, dict): the Id of the row to copy the data from and the 

    def close(self, _Frames: pd.DataFrame|None=None):
        if not self.src_tims_id is None:
            ctims.tims_close(self.src_tims_id)
            self.src_tims_id = None
        if not self.tdf_bin is None:
            self.tdf_bin.close()
            self.tdf_bin = None
        if not self.sqlcon is None:
            if len(self.frame_row_updates) > 0:
                update_frames_table(
                    frame_row_updates=self.frame_row_updates,
                    sqlite_connection=self.sqlcon,
                    _Frames=_Frames,
                )
            self.sqlcon.close()
            self.sqlcon = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def save_frame_dict(self, frame_dict, total_scans, copy_sql = True):
        if 'tof' in frame_dict:
            return self.save_frame_tofs(frame_dict['scan'], frame_dict['tof'], frame_dict['intensity'], total_scans, copy_sql)
        return self.save_frame(frame_dict['scan'], frame_dict['mz'], frame_dict['intensity'], total_scans, copy_sql)
    
    def save_frame(self, scans, mzs, intensities, total_scans, copy_sql=True):
        tofs = np.empty(len(mzs), np.double)
        if not isinstance(mzs, np.ndarray):
            mzs = np.array(mzs, np.double)
        ctims.tims_mz_to_index(self.src_tims_id, self.current_frame, cast("double*", mzs), cast("double*", tofs), len(mzs))
        return self.save_frame_tofs(scans, np.array(tofs, np.uint32), intensities, total_scans, copy_sql=copy_sql)

    def save_frame_tofs(
        self,
        scans: npt.NDArray[np.uint32],
        tofs: npt.NDArray[np.uint32],
        intensities: npt.NDArray[np.uint32],
        total_scans: int,
        copy_sql: int|bool = True,
        src_frame: int|None = None,
        run_deduplication: bool=True,
        **kwargs
    ) -> None:
        """
        Save current frame into the analysis.tdf_bin and updates the analsys.tdf.
        
        Arguments:
            src_frame (int or None): The Id of the frame in the source analysis.tdf sqlite db to copy the data from into the current frame. This duplicates the copy_sql functionality that was stupidly named by the Duke Nukem himself while he was on some kind of drugs.
            **kwargs: addiditional (column_name, value) mapping for the update of the final Frames table. 
        """
        total_scans = int(total_scans)
        if not isinstance(copy_sql, bool):# surprisingly, isinstance(True, int) is True...
            src_frame = copy_sql
        if src_frame is None:
            src_frame = self.current_frame

        frame_start_pos = int(self.tdf_bin.tell())
        if run_deduplication:
            scans, tofs, intensities = deduplicate(scans, tofs, intensities)
        assert len(scans) == len(tofs), "scans, tofs, and intensities must have the same length"
        assert len(scans) == len(intensities), "scans, tofs, and intensities must have the same length"
        num_peaks = len(scans)


        # Getting a map scan (list index) -> number of peaks
        peak_cnts = get_peak_cnts(total_scans, scans)
        if tofs.base is not None:# checking if we are dealing with a view
            tofs = np.copy(tofs)
        modify_tofs(tofs, scans)# this cannot run on a view
        if not isinstance(intensities, np.ndarray) or not intensities.dtype == np.uint32:
            intensities = np.array(intensities, np.uint32)
        interleaved = np_zip(tofs, intensities)
        real_data = get_realdata(peak_cnts, interleaved)
        compressed_data = zstd.ZSTD_compress(bytes(real_data), self.compression_level)

        self.tdf_bin.write((len(compressed_data)+8).to_bytes(4, 'little', signed = False))
        self.tdf_bin.write(total_scans.to_bytes(4, 'little', signed = False))
        self.tdf_bin.write(compressed_data)

        if copy_sql:
            frame_row_update = {
                "Id": self.current_frame,
                "TimsId": frame_start_pos,
                "NumScans": total_scans,
                "NumPeaks": num_peaks,
                "MaxIntensity": 0 if len(intensities) == 0 else int(np.max(intensities)),
                "SummedIntensities": int(np.sum(intensities)),
                "AccumulationTime": 100.0,
            }
            frame_row_update.update(kwargs)
            self.frame_row_updates.append( (src_frame, frame_row_update) )

        self.current_frame += 1




    

if __name__ == '__main__':
    from opentimspy import OpenTIMS
    ot = OpenTIMS('/mnt/storage/science/midia_data_1/midia/')
    new_path = '/mnt/storage/science/midia_data_1/midia_new/'
    shutil.rmtree(new_path)
    s = SaveTIMS(ot, '/mnt/storage/science/midia_data_1/midia_new/')
    s.save_frame([100.0, 100.0], [4, 4], [657, 657], 15)
    s.save_frame([100.0, 100.0], [4, 4], [657, 657], 15)
    s.close()
