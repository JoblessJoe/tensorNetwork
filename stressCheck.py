'''
CPU stability check with the real workload (Sprudler games on all cores) - for testing an undervolt / overclock / new CPU.

A crash is easy to see; the dangerous failure is a SILENT wrong result. The game evaluation is fully deterministic (fixed seeds,
single-threaded BLAS), so every round must give exactly the same numbers. This script plays the same fixed games again and again and
compares every round with (a) the first round and (b) scores measured earlier on another CPU (REFERENCE). Any difference = unstable.

    python stressCheck.py                        5 minutes, all cores
    python stressCheck.py --minutes 60           longer test
    python stressCheck.py --pause 600            watch mode: one short round every 10 minutes (e.g. next to a training run)

Everything is also written to models/stress_log.txt (timestamps, round times, mismatches). Round times that grow over the test
mean the CPU slows down (thermal throttling / power limit). Temperatures are not visible from WSL - watch them in HWiNFO.
'''
import argparse
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # multi-threaded BLAS changes float rounding -> not comparable
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprudelJump"))

import evalModel as ev
import proSprudler as ps

CAREFUL = "models/careful_2026-10-05_18-43-51.pt"
POLISH = "models/polishBase_2026-10-07_01-24-15.pt"
# benchmark of CAREFUL at start 0, mean per 50-game seed set (rounded), measured on the Ryzen 9 PRO 7945: independent of this CPU
REFERENCE = {(CAREFUL, "start 0"): [8144, 6824, 7817, 7155]}


COPIES = 12    # every game set is played this many times per round: keeps all workers busy (and the copies must agree with each other, too)


def makeJobs():
    '''The same games as the benchmark: both models, start 0 and hard starts (20-30k), 4 seed sets each, with their settings (stable, smoothing 0.5).'''
    sets = ev.seedSets(200)
    modes = {"start 0": (None, 0), "hard": (30000, 20000)}
    jobs = []
    for copy in range(COPIES):
        for path in (CAREFUL, POLISH):
            for label, (hi, lo) in modes.items():
                for i, seeds in enumerate(sets):
                    jobs.append(((path, label, i, copy), (path, seeds, hi, lo, 0.0, 0.5, True, 0.0, 1.0, True, 1)))
    return jobs


def runRound(pool, jobs):
    results = pool.map(ev.job, [args for _, args in jobs])    # (mean, causes) per job
    return {key: (round(res[0], 6), tuple(res[1])) for (key, _), res in zip(jobs, results)}


def main():
    ap = argparse.ArgumentParser(description="CPU stability check with the Sprudler workload")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=31)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between rounds (watch mode); 0 = continuous full load")
    a = ap.parse_args()

    log = open(os.path.join("models", "stress_log.txt"), "a")

    def note(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    jobs = makeJobs()
    note(f"stress check: {a.minutes} min, {a.workers} workers, pause {a.pause}s, {len(jobs)} jobs per round")
    first, rounds, bad, times = None, 0, 0, []
    end = time.time() + a.minutes * 60
    with Pool(a.workers, initializer=ps.workerInit, initargs=({},)) as pool:
        while time.time() < end:
            t0 = time.time()
            try:
                res = runRound(pool, jobs)
            except Exception as e:    # a crash inside a worker is also a failed test
                bad += 1
                note(f"ROUND {rounds + 1} CRASHED: {type(e).__name__}: {e}")
                rounds += 1
                continue
            dt = time.time() - t0
            rounds += 1
            times.append(dt)
            problems = []
            if first is None:
                first = res
                for key in res:    # the copies of one game set (played on different workers) must agree within the first round already
                    if res[key] != res[key[:3] + (0,)]:
                        problems.append(f"{key}: {res[key][0]} vs copy 0 {res[key[:3] + (0,)][0]}")
                for (path, label), expected in REFERENCE.items():    # absolute check of round 1 against the other CPU
                    got = [round(res[(path, label, i, 0)][0]) for i in range(len(expected))]
                    if got != expected:
                        problems.append(f"differs from the reference measured on the old CPU: {got} vs {expected}")
            else:
                for key in res:
                    if res[key] != first[key]:
                        problems.append(f"{key}: {res[key][0]} vs first round {first[key][0]}")
            if problems:
                bad += 1
                note(f"ROUND {rounds} MISMATCH ({len(problems)}): " + "; ".join(problems[:4]))
            else:
                note(f"round {rounds}: ok, {dt:.1f} s")
            if a.pause and time.time() + a.pause < end:
                time.sleep(a.pause)
    if times:
        note(f"DONE: {rounds} rounds, {bad} bad; round time min {min(times):.1f} / mean {sum(times) / len(times):.1f} / max {max(times):.1f} s; "
             f"first 5 rounds mean {sum(times[:5]) / len(times[:5]):.1f} s, last 5 rounds mean {sum(times[-5:]) / len(times[-5:]):.1f} s")
    print("RESULT:", "STABLE (no differences)" if bad == 0 else f"UNSTABLE: {bad} bad rounds - see models/stress_log.txt")


if __name__ == "__main__":
    main()
