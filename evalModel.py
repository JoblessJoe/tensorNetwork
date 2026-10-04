'''
Fixed benchmark for trained SprudelJump models, independent of how they were trained.
Every model plays the SAME levels (fixed seeds), so numbers are comparable between models and between sessions:

    python evalModel.py models/a.pt models/b.pt
    python evalModel.py models/*.pt --games 400          more games = less noise (default 200 = 4 seed sets x 50)
    python evalModel.py models/a.pt --mixed 30000        'random start' column uses start heights 0..30000 (default 30000)

Two columns: 'start 0' (a normal game) and 'random 0-N' (random start heights, like the mixed-start training;
the score counts only points earned beyond the start). 'per set' shows the mean of each 50-game seed set,
so you can see how much of a difference is just noise (identical runs differ by ~15%).

Use the env (sprudelJump checkout) the models were TRAINED on: a model trained on the old observation
sees different inputs under the screen-wrap env and scores lower there.
'''
import argparse
import os
import random
import statistics as st
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprudelJump"))
import torch
torch.set_num_threads(1)
from multiprocessing import Pool

import proSprudler as ps

SEED = 2024          # fixed benchmark: the same levels in every session (same as the ad-hoc checks so far)
SET_SIZE = 50


def seedSets(games: int):
    rng = random.Random(SEED)
    return [[rng.randrange(2**32) for _ in range(SET_SIZE)] for _ in range(games // SET_SIZE)]


def job(args):
    path, seeds, maxStart, minStart, zero, smooth = args
    net = ps.loadNetwork(path)
    return sum(ps.evaluateNetwork(net, seeds=seeds, iterations=len(seeds), maxStartHeight=maxStart, minStartHeight=minStart, zeroFraction=zero, steerSmoothing=smooth)) / len(seeds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate models on a fixed benchmark")
    ap.add_argument("models", nargs="+", help="model files (.pt)")
    ap.add_argument("--games", type=int, default=200, help="games per column, multiple of 50 (default 200)")
    ap.add_argument("--mixed", type=int, default=30000, help="max start height of the random-start column (default 30000)")
    ap.add_argument("--min-start", type=int, default=0, help="minimum start height of the random-start column (e.g. 20000 = only hard starts)")
    ap.add_argument("--ranges", nargs="+", metavar="MIN:MAX", help="start-height ranges as columns after 'start 0': MIN:MAX or MIN:MAX:ZEROSHARE, e.g. 20000:30000 20000:30000:0.5 (overrides --min-start/--mixed)")
    ap.add_argument("--smoothing", type=float, default=1.0, help="steer smoothing the models were trained with (1 = none)")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    sets = seedSets(a.games)
    ranges = []                          # (min start, max start, share of games that start at 0)
    for r in (a.ranges or [f"{a.min_start}:{a.mixed}"]):
        p = r.split(":")
        ranges.append((int(p[0]), int(p[1]), float(p[2]) if len(p) > 2 else 0.0))
    modes = [("start 0", None, 0, 0.0)] + [(f"random {lo // 1000}-{hi // 1000}k" + (f" +{int(z * 100)}%@0" if z else ""), hi, lo, z) for lo, hi, z in ranges]   # (label, max, min, zero share)

    jobs = [(m, s, h, lo, z, a.smoothing) for m in a.models for _, h, lo, z in modes for s in sets]
    with Pool(a.workers, initializer=ps.workerInit, initargs=({},)) as p:
        out = p.map(job, jobs)

    k = len(sets)
    width = max(len(os.path.basename(m)) for m in a.models)
    print(f"\n{len(sets) * SET_SIZE} games per column ({len(sets)} seed sets x {SET_SIZE}), fixed seeds\n")
    print(f"{'model':{width}}  " + "  ".join(f"{label:>34}" for label, _, _, _ in modes))
    i = 0
    for m in a.models:
        cells = []
        for _m in modes:
            v = out[i:i + k]
            i += k
            cells.append(f"{st.mean(v):7,.0f}  per set {[round(x) for x in v]}".rjust(34) if k <= 4 else f"{st.mean(v):7,.0f}  (sd {st.pstdev(v):,.0f})".rjust(34))
        print(f"{os.path.basename(m):{width}}  " + "  ".join(cells))
