'''
Exports a net for the GPU tests (sprudelJump/gpuBench.py, run in ~/venv-gpu) and the CPU reference statistics of the same games.

    python exportForGpu.py models/x.pt /tmp/x_gpu.npz [--games 200]

The npz holds the 8 weight arrays (W [1, in, out], B [1, out]) and, from fastEnv on the CPU (start 0, seeds 0..games-1, repeat 5, smoothing 0.96875):
mean score, bullets per game, kills per game, mean frames, steer change per decision, detour and deviation per 1000 frames.
'''
import argparse
import numpy as np
import proSprudler as ps
import fastEnv

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--games", type=int, default=200)
a = ap.parse_args()
net = ps.loadNetwork(a.model)
arrays = {}
for L, (w, b) in enumerate(net.layers):
    arrays[f"W{L}"] = w.detach().cpu().numpy().astype(np.float32).T[None].copy()
    arrays[f"B{L}"] = b.detach().cpu().numpy().astype(np.float32)[None].copy()
seen = {}
original = fastEnv.FastBatch.behaviour
def spy(self):
    seen["c"] = tuple(x.copy() for x in original(self)) + (self.decisions().copy(), self.hits().copy(), self.detours().copy(), self.deviations().copy())
    return seen["c"][:3]
fastEnv.FastBatch.behaviour = spy
scores = ps.evaluateNetworkFast(net, list(range(a.games)), ps.MAX_FRAMES_PER_GAME, None, 0, 0.0, 0.96875, 0.0, 1.0, 5, penalties=(1e-12, 0.0, 0.0))
shots, free, steer, decisions, hits, detour, deviation = seen["c"]
frames = decisions * 5
ref = dict(score=np.mean(scores), shots=shots.mean(), hits=hits.mean(), frames=frames.mean(), steerChange=steer.sum() / decisions.sum(),
           detour=1000 * detour.sum() / frames.sum(), deviation=1000 * deviation.sum() / frames.sum(), scoreSd=np.std(scores) / np.sqrt(len(scores)))
np.savez(a.out, **arrays, **{"ref_" + k: v for k, v in ref.items()})
print("saved", a.out, {k: round(float(v), 3) for k, v in ref.items()})
