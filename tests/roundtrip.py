import sys
import shutil

import numpy as np

from opentimspy import OpenTIMS
from savetimspy import SaveTIMS


frames_to_test = list(range(1, 1000))


if len(sys.argv) != 3:
    print("Usage:\n\n\tpython roundtrip.py <source TIMS file> <destination>\n")
    sys.exit(1)


src_tims_p = sys.argv[1]
dst_tims_p = sys.argv[2]

shutil.rmtree(dst_tims_p, ignore_errors=True)

src_tims = OpenTIMS(src_tims_p)
dst_tims = SaveTIMS(src_tims, dst_tims_p)

for frame_id in frames_to_test:
    qf = src_tims.query(frame_id)
    dst_tims.save_frame_dict(qf, 918)

dst_tims.close()

dst_tims = OpenTIMS(dst_tims_p)

for frame_id in frames_to_test:
    x1 = src_tims.query(frame_id)
    x2 = dst_tims.query(frame_id)

    #name = 'scan'
    #name = 'intensity'
    #name = 'mz'
    #print(np.array_equal(x1[name],x2[name]))
    for name in 'scan intensity mz'.split():
        assert np.allclose(x1[name],x2[name], rtol=0.0001)
print("OK!")
