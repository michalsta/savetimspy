import os
import shutil
import opentimspy
from pathlib import Path
import cffi
import sqlite3
import numpy as np
import zstd
import opentims_bruker_bridge
from savetimspy.numba_helper import (
    deduplicate,
    get_peak_cnts,
    modify_tofs,
    np_zip,
    get_realdata,    
)


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


class SaveTIMS:
    def __init__(self, opentims_obj, path, compression_level = 1):
        self.src_tims_id = None
        self.tdf_bin = None
        self.sqlcon = None
        self.srcsqlcon = None
        os.mkdir(path)
        self.opentims = opentims_obj
        self.src_path = opentims_obj.analysis_directory
        self.src_tims_id = ctims.tims_open(bytes(str(opentims_obj.analysis_directory), 'ascii'), 0)
        self.dst_path = Path(path)
        self.db_path = self.dst_path / 'analysis.tdf'
        shutil.copyfile(self.src_path / 'analysis.tdf', self.db_path)
        self.current_frame = 1
        self.tdf_bin = open(self.dst_path / 'analysis.tdf_bin', 'wb')
        self.sqlcon = sqlite3.connect(self.db_path)
        self.srcsqlcon = sqlite3.connect(self.src_path / 'analysis.tdf')
        self.sqlcon.execute("DELETE FROM Frames;")
        # self.sqlcon.execute("begin")
        self.compression_level = compression_level

    def close(self):
        if not self.src_tims_id is None:
            ctims.tims_close(self.src_tims_id)
            self.src_tims_id = None
        if not self.tdf_bin is None:
            self.tdf_bin.close()
            self.tdf_bin = None
        if not self.sqlcon is None:
            rowcount = 1
            while rowcount > 0:
                rowcount = self.sqlcon.execute("DELETE FROM Frames WHERE Id = ?", (self.current_frame,)).rowcount
                self.current_frame += 1
            self.sqlcon.commit()
            self.sqlcon.close()
            self.sqlcon = None
        if not self.srcsqlcon is None:
            self.srcsqlcon.close()
            self.srcsqlcon = None

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
    
    def save_frame(self, scans, mzs, intensities, total_scans, copy_sql = True):
        tofs = np.empty(len(mzs), np.double)
        if not isinstance(mzs, np.ndarray):
            mzs = np.array(mzs, np.double)
        ctims.tims_mz_to_index(self.src_tims_id, self.current_frame, cast("double*", mzs), cast("double*", tofs), len(mzs))
        return self.save_frame_tofs(scans, np.array(tofs, np.uint32), intensities, total_scans, copy_sql=copy_sql)

    def save_frame_tofs(
        self,
        scans,
        tofs,
        intensities,
        total_scans,
        copy_sql: int|bool = True,
        src_frame: int|None = None,
        run_deduplication: bool=True,
        set_MsMsType_to_0: bool=False
    ):
        """
        Save current frame into the analysis.tdf_bin and updates the analsys.tdf.
        
        Arguments:
            src_frame (int or None): The Id of the frame in the source analysis.tdf sqlite db to copy the data from into the current frame. This duplicates the copy_sql functionality that was stupidly named by the Duke Nukem himself while he was on some kind of drugs.
        """
        total_scans = int(total_scans)
        if copy_sql == True or isinstance(copy_sql, int):
            if src_frame is None:
                if copy_sql == True:
                    src_frame = self.current_frame
                else:
                    src_frame = copy_sql
            src_frame = int(src_frame)
            # get the src_frame info
            frame_row = list(self.srcsqlcon.execute("SELECT * FROM Frames WHERE Id == ?;", (src_frame,)))[0]
            frame_row = list(frame_row)
            frame_row[0] = self.current_frame
            qmarks = ['?'] * len(frame_row)
            qmarks = ', '.join(qmarks)
            self.sqlcon.execute("INSERT INTO Frames VALUES (" + qmarks + ")", frame_row)
            # dump the src_frame row into the new analysis.tdf
        frame_start_pos = self.tdf_bin.tell()

        if run_deduplication:
            scans, tofs, intensities = deduplicate(scans, tofs, intensities)

        # Getting a map scan (list index) -> number of peaks
        peak_cnts = get_peak_cnts(total_scans, scans)
        modify_tofs(tofs, scans)
        if not isinstance(intensities, np.ndarray) or not intensities.dtype == np.uint32:
            intensities = np.array(intensities, np.uint32)
        interleaved = np_zip(tofs, intensities)
        real_data = get_realdata(peak_cnts, interleaved)
        compressed_data = zstd.ZSTD_compress(bytes(real_data), self.compression_level)

        self.tdf_bin.write((len(compressed_data)+8).to_bytes(4, 'little', signed = False))
        self.tdf_bin.write(total_scans.to_bytes(4, 'little', signed = False))
        self.tdf_bin.write(compressed_data)

        sql_input = (
            total_scans,
            len(tofs),
            frame_start_pos,
            0 if len(intensities) == 0
            else int(np.max(intensities)),
            int(np.sum(intensities)),
            self.current_frame
        )
        if not set_MsMsType_to_0:
            sql = """UPDATE Frames SET
                NumScans = ?,
                NumPeaks = ?,
                TimsId = ?,
                MaxIntensity = ?,
                SummedIntensities = ?,
                AccumulationTime = 100.0
            WHERE
                Id = ?;"""
        else:
            sql = """UPDATE Frames SET
                MsMsType = ?,
                NumScans = ?,
                NumPeaks = ?,
                TimsId = ?,
                MaxIntensity = ?,
                SummedIntensities = ?,
                AccumulationTime = 100.0
            WHERE
                Id = ?;"""
            sql_input = (0,) + sql_input
      
        F = self.sqlcon.execute(sql, sql_input).rowcount
        self.sqlcon.commit()
        self.current_frame += 1

        return



    

if __name__ == '__main__':
    from opentimspy import OpenTIMS
    ot = OpenTIMS('/mnt/storage/science/midia_data_1/midia/')
    new_path = '/mnt/storage/science/midia_data_1/midia_new/'
    shutil.rmtree(new_path)
    s = SaveTIMS(ot, '/mnt/storage/science/midia_data_1/midia_new/')
    s.save_frame([100.0, 100.0], [4, 4], [657, 657], 15)
    s.save_frame([100.0, 100.0], [4, 4], [657, 657], 15)
    s.close()
