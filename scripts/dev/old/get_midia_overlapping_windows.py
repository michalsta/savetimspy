import pathlib
from savetimspy.merging_groups import save_moving_k_mer_MIDIA_diagonals


def main():
    import argparse
    cmdpar = argparse.ArgumentParser(description="Extract data from tdf MIDIA scans and turn them int MS1 tdfs.")
    cmdpar.add_argument(
        "source_tdf_folder",
        help="Path to TimsTOF Pro .d folder.",
        type=pathlib.Path,
    )
    cmdpar.add_argument(
        "destination_folder",
        help="Path to target folder where subfolders with individual MS2 signals will be dumped.",
        type=pathlib.Path,
    )
    cmdpar.add_argument(
        "--group_overlap",
        default=1,
        type=int,
        help="The overlap between MS2 diagonals."
    )
    cmdpar.add_argument(
        "--verbose",
        help="Show progress bars.",
        action="store_true",
    )
    kwargs = cmdpar.parse_args().__dict__
    save_moving_k_mer_MIDIA_diagonals(**kwargs)


if __name__ == "__main__":
    main()
