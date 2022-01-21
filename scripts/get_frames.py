import sys
import sqlite3
from pathlib import Path
import shutil
from opentimspy import OpenTIMS
from savetimspy import SaveTIMS


import argparse
parser = argparse.ArgumentParser(description='Extract a set of frames from a TDF dataset.')
parser.add_argument("in_p", help="source path")
parser.add_argument("out_p", help="destination path")
parser.add_argument("frames", help="comma-separated list of frames to extract. May contain ranges. Example: 314,317-320,350")
parser.add_argument("--force", "-f", help="force overwriting of the target path if it exists", action='store_true')
parser.add_argument("--ms1", help="mark all frames as ms1", action='store_true')
args = parser.parse_args()

src = Path(args.in_p)
dst = Path(args.out_p)

frames = set()
for frame_desc in args.frames.split(','):
    print(frame_desc)
    if '-' in frame_desc:
        start, end = frame_desc.split('-')
        print(start, end)
        frames.update(range(int(start), int(end)+1))
        print(frames)
    else:
        frames.add(int(frame_desc))
frames = sorted(frames)

if args.force:
    shutil.rmtree(dst, ignore_errors=True)

assert not dst.exists()

with OpenTIMS(src) as ot, SaveTIMS(ot, dst) as s, sqlite3.connect(src / 'analysis.tdf') as db:
    for frame in frames:
        D = ot.query(frame, columns = "scan tof intensity".split())
        n_scans = list(db.execute("SELECT NumScans FROM Frames WHERE Id == ?", (frame,)))[0][0]
        s.save_frame_tofs(D['scan'], D['tof'], D['intensity'], n_scans)

if args.ms1:
    with sqlite3.connect(dst / 'analysis.tdf'):
        db.execute("UPDATE Frames set MsMsType=0;")
