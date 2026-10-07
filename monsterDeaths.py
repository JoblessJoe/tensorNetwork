'''
Why does the network die to monsters while RISING? Replays fixed games of a model (fast env, same settings as the benchmark) and, for every
death by a monster while rising (cause 1), looks back at the last frames before the hit:

  - where was the killer monster relative to the player (horizontal distance dx in px, vertical distance dy in px) 45 / 30 / 20 / 10 frames before?
  - was the player already on a collision course (|dx| < 38 px = half player width + half monster width)?
  - did the player steer TOWARD the monster or away from it in the last 20 frames, and did it shoot?

    python monsterDeaths.py models/polishBase_2026-10-07_01-24-15.pt            start 0 and hard starts (20-30k), 200 games each
    python monsterDeaths.py models/x.pt --games 400 --stable 3,2,1,2

Reading it: if most deaths are already on a collision course 45 frames before the hit and the net steered toward the monster, the policy walks into
monsters it could dodge (an indecision / learning problem). If the monster only comes into reach in the last ~10 frames, the death was forced by the
platform layout (a jump that cannot also avoid the monster).
'''
import argparse
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprudelJump"))

import numpy as np

import evalModel as ev
import proSprudler as ps
from fastEnv import FastBatch, PX, PY, VY, NM

HIST = 64                # frames of history per game
PLAYER_W, MONSTER_W, MONSTER_H, SCREEN_W = 40, 36, 36, 400
SCROLL_THRESHOLD, GRAVITY, MAX_FALL = 280.0, 0.4, 15.0    # env constants (SCROLL_THRESHOLD_Y = 0.4 * 700)
COLLISION_DX = (PLAYER_W + MONSTER_W) / 2     # 38 px: closer than this horizontally = on collision course


def playGames(net, seeds, maxStart, minStart, slots, smoothing=0.5, maxFrames=100_000):
    '''Plays the games like evaluateNetworkFast and returns one record per game that ended by a monster while rising.'''
    layers = [(w.detach().cpu().numpy().astype(np.float32).T.copy(), b.detach().cpu().numpy().astype(np.float32)) for w, b in net.layers]
    nIn = layers[0][0].shape[0]
    batch = FastBatch(seeds, maxStart, minStart, 0.0, 0.0, 1.0, stable=slots)
    G = len(seeds)
    hist = np.zeros((G, HIST, 6))    # per frame BEFORE the step: playerX, playerY, velY, scrolled px so far, applied steer, shoot
    scrolled = np.zeros(G)           # total px the screen has scrolled (monsters keep screen y - scrolled constant; stomp bonuses are NOT scroll)
    frames = np.zeros(G, dtype=np.int64)
    applied = np.full(G, 0.5)
    alive = np.arange(G, dtype=np.int64)
    records, causes = [], np.zeros(G, dtype=np.int64)
    while alive.size and frames.max() < maxFrames:
        x = batch.observe(alive, nIn).astype(np.float32)
        for L, (wT, b) in enumerate(layers):
            x = x @ wT + b
            x = np.maximum(x, 0) if L < len(layers) - 1 else 1 / (1 + np.exp(-x))
        steer = x[:, 0].astype(np.float64)
        if smoothing < 1.0:
            applied[alive] += smoothing * (steer - applied[alive])
            steer = applied[alive]
        slot = frames[alive] % HIST
        hist[alive, slot] = np.stack([batch.F[alive, PX], batch.F[alive, PY], batch.F[alive, VY], scrolled[alive], steer, x[:, 1]], axis=1)
        pyPre, vyPre = batch.F[alive, PY].copy(), batch.F[alive, VY].copy()
        batch.step(alive, steer, x[:, 1].astype(np.float64), 1)
        scrolled[alive] += np.maximum(0.0, SCROLL_THRESHOLD - (pyPre + np.minimum(vyPre + GRAVITY, MAX_FALL)))
        frames[alive] += 1
        for g in alive[batch.done()[alive]]:
            causes[g] = batch.causes()[g]
            if causes[g] == 1:
                rec = killerRecord(batch, hist, frames, scrolled, g)
                if rec is not None:
                    records.append(rec)
        alive = alive[~batch.done()[alive]]
    return records, causes


def killerRecord(batch, hist, frames, scrolled, g):
    '''Finds the monster that overlaps the player after the fatal step and its relative position in the frames before.'''
    px, py = batch.F[g, PX], batch.F[g, PY]
    killer = None
    for i in range(batch.I[g, NM]):
        mx, my = batch.MO[g, i]
        if px < mx + MONSTER_W and px + PLAYER_W > mx and py < my + MONSTER_H and py + 40 > my:
            killer = (mx, my - scrolled[g])     # x, 'world y': screen y minus the scroll so far stays constant
            break
    if killer is None:
        return None
    last = frames[g] - 1
    rec = {}
    for k in (45, 30, 20, 10):
        n = last - k
        if n < 0 or k >= HIST:
            rec[k] = None
            continue
        pxk, pyk, vyk, totk, _, _ = hist[g, n % HIST]
        dx = (killer[0] + MONSTER_W / 2) - (pxk + PLAYER_W / 2)
        dx = (dx + SCREEN_W / 2) % SCREEN_W - SCREEN_W / 2            # shorter way around the screen edge
        rec[k] = (dx, (killer[1] + totk) - pyk)                       # horizontal / vertical distance (monster top minus player top; < 0 = monster above the player), px
    # steering over the last 20 frames relative to the monster's direction 20 frames before the hit
    if rec[20] is not None:
        sign = np.sign(rec[20][0]) or 1.0
        window = [hist[g, (last - j) % HIST] for j in range(20) if last - j >= 0]
        rec["toward"] = float(np.mean([(w[4] - 0.5) * sign for w in window]))       # > 0 = steered toward the monster
        rec["shoot"] = float(np.mean([w[5] > 0.5 for w in window]))
    return rec


def report(records, causes, label):
    n = len(causes)
    names = {1: "monster while rising", 2: "monster, other", 3: "fell", 4: "no progress"}
    print(f"\n=== {label}: {n} games ===")
    print("deaths: " + ", ".join(f"{names[c]} {100 * np.sum(causes == c) / n:.0f}%" for c in (1, 2, 3, 4)))
    print(f"monster-while-rising analysed: {len(records)} games")
    for k in (45, 30, 20, 10):
        vals = [r[k] for r in records if r[k] is not None]
        if not vals:
            continue
        dx = np.abs([v[0] for v in vals])
        dy = np.array([v[1] for v in vals])
        print(f"  {k:2d} frames before: median |dx| {np.median(dx):5.0f} px, on collision course (|dx| < {COLLISION_DX:.0f}) {100 * np.mean(dx < COLLISION_DX):3.0f}%, "
              f"median dy (monster top - player top) {np.median(dy):5.0f} px   (n={len(vals)})")
    toward = [r["toward"] for r in records if "toward" in r]
    shoot = [r["shoot"] for r in records if "shoot" in r]
    if toward:
        print(f"  last 20 frames: mean steer toward the monster {np.mean(toward):+.3f} (0 = neutral, > 0 toward, max +-0.5), "
              f"steered toward in {100 * np.mean(np.array(toward) > 0):.0f}% of the deaths; shooting in {100 * np.mean(shoot):.0f}% of the frames")


def main():
    ap = argparse.ArgumentParser(description="Analyse monster deaths while rising")
    ap.add_argument("model")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--stable", default="3,2,1,2", help="stable slot counts the model was trained with (default 3,2,1,2)")
    ap.add_argument("--smoothing", type=float, default=0.5)
    a = ap.parse_args()
    net = ps.loadNetwork(a.model)
    slots = tuple(int(v) for v in a.stable.split(","))
    seeds = [s for group in ev.seedSets(a.games) for s in group]
    for label, hi, lo in (("start 0", None, 0), ("hard starts 20-30k", 30000, 20000)):
        records, causes = playGames(net, seeds, hi, lo, slots, a.smoothing)
        report(records, causes, label)


if __name__ == "__main__":
    main()
