import sys
import sqlite3
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
from collections import Counter, defaultdict


parser = argparse.ArgumentParser(description="Flatten a group of scans")
parser.add_argument("src", metavar="<source.d>", type=Path, help="Source TDF directory")
parser.add_argument("dst", metavar="<destination.d>", type=Path, help="Output directory")
parser.add_argument("-s", "--silent", action="store_true", help="Silent (do not show progressbar)")
parser.add_argument("-f", "--force", action="store_true", help="Delete the target directory if it exists")
args = parser.parse_args()

from opentimspy import OpenTIMS
from savetimspy import SaveTIMS


assert args.src.is_dir()

progressbar = (lambda x: x) if args.silent else tqdm

ot = OpenTIMS(args.src)

if args.force:
    shutil.rmtree(args.dst, ignore_errors=True)

s = SaveTIMS(ot, args.dst)
db = sqlite3.connect(args.src / 'analysis.tdf')



groups = {}
for frame, win_gr in db.execute("SELECT Frame, WindowGroup FROM DiaFrameMsMsInfo;"):
    groups[int(frame)] = int(win_gr)

frame_to_scans = {}
for frame, scans in db.execute("SELECT Id, NumScans FROM Frames;"):
    frame_to_scans[int(frame)] = int(scans)

max_frame = max(groups.keys())

collected = None

for frame_id in progressbar(range(1, max_frame+2)):
    if not frame_id in groups:
        if collected is not None:
            collected.sort()
            scan, tof, intens = zip(*collected)
            s.save_frame_tofs(scan, tof, intens, frame_to_scans[frame_id-1], copy_sql=ms1_id)
            pass
        ms1_id = frame_id
        collected = []
    else:
        D = ot.query(frame_id, columns="scan tof intensity".split())
        n_scans = frame_to_scans[frame_id]
        collected.extend(zip(D['scan'], D['tof'], D['intensity']))


