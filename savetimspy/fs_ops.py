
def reset_max_open_soft_file_handles(
    min_soft: int=4_096,
    multiplier: float=4.0,
    verbose: bool=False,
) -> None:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = max(min_soft, int(multiplier*soft))
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    if verbose:
        import subprocess
        print(subprocess.check_output("whoami; ulimit -n", shell=True))

