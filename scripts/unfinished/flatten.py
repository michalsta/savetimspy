import sys
import sqlite3
import argparse
from pathlib import Path
import shutil
from opentimspy import OpenTIMS
from savetimspy import SaveTIMS
from collections import Counter


parser = argparse.ArgumentParser(description="Flatten a group of scans")
parser.add_argument("src", metavar="<source.d>", type=Path)
parser.add_argument("dst", metavar="<destination.d>", type=Path)
parser.add_argument("group", metavar="<group id>", type=int)
parser.parse_args()

if len(sys.argv) != 4:
    print(
'''Usage:
        python get_midia_groups.py <source.d> <destination.d> <group id to obtain>
''')
    sys.exit(1)

src = parser.src
dst = parser.dst
group = parser.group
ot = OpenTIMS(src)
#shutil.rmtree(dst)
s = SaveTIMS(ot, dst)
db = sqlite3.connect(src / 'analysis.tdf')

max_window_group = list(db.execute("SELECT MAX(WindowGroup) FROM DiaFrameMsMsInfo;"))[0][0]
print(max_window_group)

S = defaultdict(float)
n_scans = 0
for frame_id, window_grp in list(db.execute("SELECT Frame, WindowGroup FROM DiaFrameMsMsInfo;")):
    D = ot.query(frame_id)
    n_scans = max(m_scans, list(db.execute("SELECT NumScans FROM Frames WHERE Id == ?", (frame_id,)))[0][0])
    for mass, scan, intensity in zip(D['mz'], D['scan'], D['intensity']):
        S[(mass, scan)] += intensity

    if window_grp == max_window_group:
#        L = 
        s.save_frame(D['mz'], D['scan'], D['intensity'], n_scans)

