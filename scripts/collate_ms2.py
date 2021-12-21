import sys
import sqlite3
import argparse
from pathlib import Path
import shutil
from collections import Counter, defaultdict


parser = argparse.ArgumentParser(description="Flatten a group of scans")
parser.add_argument("src", metavar="<source.d>", type=Path)
parser.add_argument("dst", metavar="<destination.d>", type=Path)
args = parser.parse_args()

from opentimspy import OpenTIMS
from savetimspy import SaveTIMS


assert args.src.is_dir()


ot = OpenTIMS(args.src)
shutil.rmtree(args.dst)
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

for frame_id in range(1, max_frame+2):
    if not frame_id in groups:
        if collected is not None:
            collected.sort()
            scan, mz, intens = zip(*collected)
            s.save_frame(scan, mz, intens, frame_to_scans[frame_id-1])
            pass
        collected = []
    else:
        D = ot.query(frame_id)
        n_scans = frame_to_scans[frame_id]
        collected.extend(zip(D['scan'], D['mz'], D['intensity']))



