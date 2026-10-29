'''
Watches a running training run: every new snapshot (<name>_..._g<gen>.pt) gets the start-0 benchmark (evalModel.py's fixed 200 games).
When one reaches the target, the run gets a Ctrl-C (SIGINT): the ES loop ends the phase cleanly, saves net + pool, nightRun benchmarks as usual.

    python targetWatch.py --name longArcGpu2 --pid <nightRun pid> [--target 100000] [--workers 4]

Results are appended to models/target_<name>.txt. The watcher ends by itself when the run's process is gone.
'''
import argparse
import glob
import os
import re
import signal
import statistics as st
import time
from datetime import datetime
from multiprocessing import Pool

import evalModel as em
import proSprudler as ps


def log(path, text):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def snapGen(path):
    return int(re.search(r"_g(\d+)\.pt$", path).group(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark every new snapshot of a run, stop the run at a target score (start 0)")
    ap.add_argument("--name", required=True, help="phase name prefix of the run's files in models/ (e.g. longArcGpu2)")
    ap.add_argument("--pid", type=int, required=True, help="pid of the run's main nightRun.py process (gets SIGINT at the target)")
    ap.add_argument("--target", type=float, default=100000, help="start-0 benchmark mean that stops the run (default 100000)")
    ap.add_argument("--games", type=int, default=200, help="games of the benchmark, multiple of 50 (default 200, same seeds as evalModel.py)")
    ap.add_argument("--smoothing", type=float, default=0.96875)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4, help="the benchmark has one job per 50-game seed set, so more than games/50 workers do nothing")
    ap.add_argument("--poll", type=float, default=60, help="seconds between looks for new snapshots")
    a = ap.parse_args()

    logPath = os.path.join("models", f"target_{a.name}.txt")
    sets = em.seedSets(a.games)
    done = set()
    log(logPath, f"watching models/{a.name}_*_g*.pt of pid {a.pid}, target start 0 >= {a.target:,.0f} ({a.games} games)")
    with Pool(a.workers, initializer=ps.workerInit, initargs=({},)) as pool:
        while alive(a.pid):
            ready = [p for p in glob.glob(os.path.join("models", f"{a.name}_*_g*.pt"))
                     if p not in done and time.time() - os.path.getmtime(p) > 10]    # 10 s old: the save is finished
            for path in sorted(ready, key=snapGen):
                done.add(path)
                jobs = [(path, s, None, 0, 0.0, a.smoothing, 0.0, 1.0, True, a.repeat) for s in sets]    # start 0 column of evalModel.py
                perSet = [o[0] for o in pool.map(em.job, jobs)]
                score = st.mean(perSet)
                log(logPath, f"{os.path.basename(path)}: start 0 {score:,.0f}  per set {[round(x) for x in perSet]}")
                if score >= a.target and alive(a.pid):
                    log(logPath, f"target reached -> SIGINT to pid {a.pid} (run ends the phase, saves, benchmarks)")
                    os.kill(a.pid, signal.SIGINT)
                    raise SystemExit
            time.sleep(a.poll)
    log(logPath, "run process is gone, watcher ends")
