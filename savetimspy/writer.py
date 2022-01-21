import os
import shutil
from pathlib import Path
import cffi
import sqlite3
import numpy as np
import zstd
import opentims_bruker_bridge


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
            return self.save_frame_tof(frame_dict['scan'], frame_dict['tof'], frame_dict['intensity'], total_scans, copy_sql)
        return self.save_frame(frame_dict['scan'], frame_dict['mz'], frame_dict['intensity'], total_scans, copy_sql)
    
    def save_frame(self, scans, mzs, intensities, total_scans, copy_sql = True):
        tofs = np.empty(len(mzs), np.double)
        if not isinstance(mzs, np.ndarray):
            mzs = np.array(mzs, np.double)
        ctims.tims_mz_to_index(self.src_tims_id, self.current_frame, cast("double*", mzs), cast("double*", tofs), len(mzs))
        return self.save_frame_tofs(scans, np.array(tofs, np.uint32), intensities, total_scans, copy_sql=copy_sql)

    def save_frame_tofs(self, scans, tofs, intensities, total_scans, copy_sql = True):
        if copy_sql == True or isinstance(copy_sql, int):
            if copy_sql == True:
                src_frame = self.current_frame
            else:
                src_frame = copy_sql
            frame_row = list(self.srcsqlcon.execute("SELECT * FROM Frames WHERE Id == ?;", (src_frame,)))[0]
            frame_row = list(frame_row)
            frame_row[0] = self.current_frame
            qmarks = ['?'] * len(frame_row)
            qmarks = ', '.join(qmarks)
            self.sqlcon.execute("INSERT INTO Frames VALUES (" + qmarks + ")", frame_row)
        frame_start_pos = self.tdf_bin.tell()
        if not isinstance(tofs, np.ndarray):
            tofs = np.array(tofs, np.uint32)

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

        if not isinstance(intensities, np.ndarray) or not intensities.dtype == np.uint32:
            intensities = np.array(intensities, np.uint32)

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

        compressed_data = zstd.ZSTD_compress(bytes(real_data), self.compression_level)

        self.tdf_bin.write((len(compressed_data)+8).to_bytes(4, 'little', signed = False))
        self.tdf_bin.write(total_scans.to_bytes(4, 'little', signed = False))
        self.tdf_bin.write(compressed_data)

        F = self.sqlcon.execute("UPDATE Frames SET NumScans = ?, NumPeaks = ?, TimsId = ?, AccumulationTime = 100.0 WHERE Id = ?;", (total_scans, len(tofs), frame_start_pos, self.current_frame)).rowcount
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
