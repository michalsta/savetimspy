import sys
import sqlite3
from pathlib import Path
import shutil
from opentimspy import OpenTIMS
from savetimspy import SaveTIMS

if len(sys.argv) != 4:
    print(
'''Usage:
        python get_midia_groups.py <source.d> <destination.d> <frame id to obtain>
''')
    sys.exit(1)

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
frame = int(sys.argv[3])
ot = OpenTIMS(src)
shutil.rmtree(dst)
s = SaveTIMS(ot, dst)
db = sqlite3.connect(src / 'analysis.tdf')

D = ot.query(frame)
n_scans = list(db.execute("SELECT NumScans FROM Frames WHERE Id == ?", (frame,)))[0][0]
s.save_frame(D['scan'], D['mz'], D['intensity'], n_scans)

