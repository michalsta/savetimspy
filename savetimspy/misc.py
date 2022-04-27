

class FullSet():
    def __init__(self):
        pass
    def __contains__(self, what):
        return True


def int_set(description):
    '''Parses strings like "23,45,47-51,55" into sets:
    set(23, 45, 47, 48, 49, 50, 51, 55)'''
    if description is None:
        return FullSet()

    ret = set()
    for desc in description.split(","):
        if '-' in desc:
            start, end = desc.split('-')
            ret.update(range(int(start), int(end)+1))
        else:
            ret.add(int(desc))

    return ret
