


def bruker_encode(in_bytes):
    n = len(in_bytes)
    out = bytearray(n)
    i = 0
    j = 0
    while True:
        out[i] = in_bytes[j]
        i += 1
        j += 4
        if j >= n:
            j = j - n + 1
            if j == 4:
                break
    return out


def bruker_decode(in_bytes):
    n = len(in_bytes)
    out = bytearray(n)
    i = 0
    j = 0
    while True:
        out[j] = in_bytes[i]
        i += 1
        j += 4
        if j >= n:
            j = j - n + 1
            if j == 4:
                break
    return out


def _check(b):
    assert bruker_encode(bruker_decode(b)) == b
    assert bruker_decode(bruker_encode(b)) == b
