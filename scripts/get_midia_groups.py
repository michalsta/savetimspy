import sys
import sqlite3
from pathlib import Path
import shutil
from opentimspy import OpenTIMS
from savetimspy import SaveTIMS

if len(sys.argv) != 4:
    print(
'''Usage:
        python get_midia_groups.py <source.d> <destination.d> <group id to obtain>
''')
    sys.exit(1)

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
group = int(sys.argv[3])
ot = OpenTIMS(src)
shutil.rmtree(dst)
s = SaveTIMS(ot, dst)
db = sqlite3.connect(src / 'analysis.tdf')

for frame_id in list(db.execute("SELECT Frame FROM DiaFrameMsMsInfo WHERE WindowGroup == ?;", (group,))):
    frame_id = frame_id[0]
    D = ot.query(frame_id)
    n_scans = list(db.execute("SELECT NumScans FROM Frames WHERE Id == ?", (frame_id,)))[0][0]
    s.save_frame(D['mz'], D['scan'], D['intensity'], n_scans)

