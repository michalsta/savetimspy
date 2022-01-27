#!/usr/bin/env python
import sys
import sqlite3
import argparse
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from opentimspy import OpenTIMS
from savetimspy import SaveTIMS


parser = argparse.ArgumentParser(description="Flatten a series MS2 MIDIA scans into MS1. Frames from each MS2 cycle are summed up into a single MS1 frame.")
parser.add_argument("src", metavar="<source.d>", type=Path, help="Source TDF directory")
parser.add_argument("dst", metavar="<destination.d>", type=Path, help="Output directory")
parser.add_argument("-s", "--silent", action="store_true", help="Silent (do not show progressbar)")
parser.add_argument("-f", "--force", action="store_true", help="Delete the target directory if it exists")
parser.add_argument("-c", "--cycles", help="Comma-separated list MIDIA cycles or cycle ranges to extract. Example: 314,340-350,356. Extract everything if omitted.", default="")
args = parser.parse_args()

assert args.src.is_dir()

cycles = set()
for cycle_desc in args.cycles.split(","):
    if '-' in cycle_desc:
        start, end = cycle_desc.split('-')
        cycles.update(range(int(start), int(end)+1))
    else:
        cycles.add(int(cycle_desc))

def extract_cycle(cycle_id):
    if len(cycles) == 0:
        return True
    return cycle_id in cycles

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

ms1_id = -1
cycle_id = 0

if not args.silent:
    total_count = max_frame+1 if len(cycles)==0 else len(cycles)
    progress_tqdm = tqdm(total=total_count)

for frame_id in range(1, max_frame+2):
    if not frame_id in groups:
        if collected is not None:
            if extract_cycle(cycle_id):
                scan, tof, intens = zip(*collected)
                scan = np.concatenate(scan)
                tof = np.concatenate(tof)
                intens = np.concatenate(intens)
                s.save_frame_tofs(scan, tof, intens, frame_to_scans[frame_id-1], copy_sql=ms1_id)
                if not args.silent:
                    progress_tqdm.update()
            cycle_id += 1
        ms1_id = frame_id
        collected = []
    else:
        if extract_cycle(cycle_id):
            D = ot.query(frame_id, columns="scan tof intensity".split())
            n_scans = frame_to_scans[frame_id]
            collected.append((D['scan'], D['tof'], D['intensity']))
