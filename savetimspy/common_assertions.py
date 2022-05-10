import pathlib


def assert_minimal_input_for_clusterings_exist(path: pathlib.Path):
    assert path.exists(), f"File {path} does not exist."
    tdf = path/"analysis.tdf"
    assert tdf.exists(), f"File {tdf} does not exist."
    tdf_bin = path/"analysis.tdf_bin"
    assert tdf_bin.exists(), f"File {tdf_bin} does not exist."