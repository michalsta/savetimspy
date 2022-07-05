from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Extract a given MIDIA group from a TDF file.")
parser.add_argument("src", metavar="<source.d>", type=Path, help="Source TDF directory")
parser.add_argument("dst", metavar="<destination.d>", type=Path, help="Output directory")
#parser.add_argument("-s", "--silent", action="store_true", help="Silent (do not show progressbar)")
parser.add_argument("-f", "--force", action="store_true", help="Delete the target directory if it exists")
#parser.add_argument("-c", "--cycles", help="Comma-separated list MIDIA cycles or cycle ranges to extract. Example: 314,340-350,356. Extract everything if omitted.", type=int_set, default=int_set(None))
parser.add_argument("-g", "--group-id", type=int, help="MIDIA group to extract (usually in range 0-19")
args = parser.parse_args()


import sys
import sqlite3
import shutil
from opentimspy import OpenTIMS
from savetimspy import SaveTIMS
from tqdm import tqdm

ot = OpenTIMS(src)
db = sqlite3.connect(src / 'analysis.tdf')
if args.force:
    shutil.rmtree(dst)
s = SaveTIMS(ot, dst)


for frame_id in list(db.execute("SELECT Frame FROM DiaFrameMsMsInfo WHERE WindowGroup == ?;", (args.group_id+1,))):
    frame_id = frame_id[0]
    D = ot.query(frame_id, columns='scan tof intensity'.split())
    n_scans = list(db.execute("SELECT NumScans FROM Frames WHERE Id == ?", (frame_id,)))[0][0]
    s.save_frame_dict(D, n_scans)




