'''
How does a net behave? Plays fixed start-0 games and prints, per model: score, game length, bullets fired per 100 frames (the cooldown allows 10),
share of the bullets fired with no monster on screen, and the mean absolute steer change per decision (0 = steady, 1 = full swing every decision).

    python behaviour.py models/a.pt models/b.pt [--games 30] [--repeat 5] [--smoothing 0.96875] [--slots 3,2,1,2,0,1]
'''
import argparse
import numpy as np
import proSprudler as ps
import fastEnv

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--games", type=int, default=30)
ap.add_argument("--repeat", type=int, default=5)
ap.add_argument("--smoothing", type=float, default=0.96875)
ap.add_argument("--slots", default="3,2,1,2,0,1")
a = ap.parse_args()

seen = {}
original = fastEnv.FastBatch.behaviour
def spy(self):    # evaluateNetworkFast only returns scores: read the counters of its batch on the way
    seen["counters"] = tuple(x.copy() for x in original(self)) + (self.decisions().copy(),)
    return seen["counters"][:3]
fastEnv.FastBatch.behaviour = spy

print(f"{a.games} games from start 0, seeds 0-{a.games - 1}, repeat {a.repeat}, smoothing {a.smoothing}")
print(f"{'model':62s} {'score':>8s} {'frames':>8s} {'bullets/100f':>13s} {'free share':>11s} {'steer chg/dec':>14s}")
for path in a.models:
    net = ps.loadNetwork(path)
    scores = ps.evaluateNetworkFast(net, list(range(a.games)), ps.MAX_FRAMES_PER_GAME, None, 0, 0.0, a.smoothing, tuple(int(x) for x in a.slots.split(",")),
                                    0.0, 1.0, a.repeat, penalties=(1e-12, 0.0, 0.0))    # a tiny penalty only so that the counters are read
    shots, free, steer, decisions = seen["counters"]
    frames = decisions * a.repeat
    print(f"{path.split('/')[-1][:62]:62s} {np.mean(scores):8,.0f} {frames.mean():8,.0f} {100 * shots.sum() / frames.sum():13.2f} "
          f"{free.sum() / max(1.0, shots.sum()):11.0%} {steer.sum() / decisions.sum():14.3f}")
