# first filter on number of solids. we don't want assemblies
# a script is kept only if it runs and its `result` is exactly 1 solid
# usage: set SPLIT below, then from scripts/: python data_filtration/filter_data.py
# (goes one script at a time, so train takes many hours. we ran 4 copies in parallel to finish sooner)

import os, time, shutil
import multiprocessing as mp
import cadquery

SPLIT = "train"
# SPLIT = "val"
# SPLIT = "test"

source = f"../data/{SPLIT}/"
destination = f"../filtered_data/filtered_{SPLIT}/"

TIMEOUT = 30        # seconds per script
MAX_RAM_MB = 1500   # laptop has 7 GB and a few scripts eat a lot of ram


# runs in the child process: run the script and count the solids in result
def count_solids(program, conn):
    # hide anything the script prints
    # devnull = os.open(os.devnull, os.O_WRONLY)
    # os.dup2(devnull, 1)
    # os.dup2(devnull, 2)
    try:
        code = {}
        exec(program, code)
        conn.send(len(code["result"].solids().vals()))
    except Exception as e:
        conn.send(type(e).__name__)

# ram used by a process in MB
def ram_mb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return 0

# run the script in its own process, so we can kill it if it hangs or eats too much ram
# returns the number of solids, or a word saying what went wrong
def run_program(program):
    parent, child = mp.Pipe()
    p = mp.Process(target=count_solids, args=(program, child))
    p.start()
    start = time.time()
    result = None
    while p.is_alive():
        if time.time() - start > TIMEOUT:
            result = "timeout"
            break
        if ram_mb(p.pid) > MAX_RAM_MB:
            result = "too_much_memory"
            break
        p.join(0.2)
    if p.is_alive():
        p.kill()
    p.join()
    if result is None:
        result = parent.recv() if parent.poll() else "crash"
    return result


if __name__ == "__main__":
    os.makedirs(destination, exist_ok=True)
    files = sorted(os.listdir(source))

    kept = 0
    for i, file in enumerate(files):
        with open(source + file) as f:
            program = f.read()
        if run_program(program) == 1:
            shutil.copy(source + file, destination)
            kept += 1
        if i % 500 == 0:
            print(f"{i}/{len(files)} scripts done, {kept} kept", flush=True)

    print(f"kept {kept} of {len(files)} scripts in {destination}")
