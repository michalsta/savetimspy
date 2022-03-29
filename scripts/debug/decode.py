import zstd
import sys
import struct
import numpy as np
from savetimspy.ops import bruker_decode, _check, bruker_encode


with open(sys.argv[1], "rb") as f:
    x = f.read(4)
    size = struct.unpack("I", x)[0]
    x = f.read(4)
    #print(struct.unpack("I", x)[0])
    comp = f.read(size-8)
    buf = zstd.ZSTD_uncompress(comp)
    ba = np.frombuffer(bruker_decode(buf), dtype=np.uint32)
    #ba = np.frombuffer(buf, dtype=np.uint32)

    for i in range(0, len(ba), 2):
        print(ba[i], ba[i+1])
