import resource
import subprocess

def reset_max_open_soft_file_handles(
    min_soft: int=4_096,
    multiplier: float=4.0,
    verbose: bool=False,
) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = max(min_soft, int(multiplier*soft))
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    if verbose: 
        print(subprocess.check_output("whoami; ulimit -n", shell=True))


def get_limits():
    return resource.getrlimit(resource.RLIMIT_NOFILE)

def set_soft_limit(
    soft: int=4096
):
    old_soft, hard = get_limits()
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

def print_limits():
    print(subprocess.check_output("whoami; ulimit -n", shell=True))
