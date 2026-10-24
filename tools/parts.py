"""Where do the ~40 us per decision step go inside the fused env call? Real mid-game states of rep10L, timed in isolation."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import proSprudler as ps
import fastEnv as fe
from fastEnv import FastBatch
LAYOUT = (3, 2, 1, 2, 0, 1)
net = ps.loadNetwork("models/rep10L_2026-10-24_02-52-46.pt")
layers = [(w.detach().numpy().astype(np.float32).T.copy(), b.detach().numpy().astype(np.float32)) for w, b in net.layers]
seeds = list(range(5000, 5200))
b = FastBatch(seeds, None, 0, 0.0, 0.0, 1.0, stable=LAYOUT)
alive, obs = b.startLoop(33)
n = len(alive); applied = np.full(200, 0.5)
smooth = 1 - 0.5 ** 10
for step in range(400):            # play 400 decisions (4000 frames), then freeze a representative state
    x = obs[:n]
    for L, (wT, bb) in enumerate(layers):
        x = x @ wT + bb
        x = np.maximum(x, 0) if L < len(layers) - 1 else 1 / (1 + np.exp(-x))
    n = b.stepObserve(alive, n, x, applied, smooth, 10)
print("alive games in the frozen state:", n)
F0, I0, PL0, MO0, BU0, MT0 = (a.copy() for a in (b.F, b.I, b.PL, b.MO, b.BU, b.MT))
al = alive[:n].copy()
mode = b.loopMode
out32 = np.zeros((200, 33), np.float32); out64 = np.zeros((200, 33))
y = np.full((n, 2), 0.6, np.float32)
def timeit(f, reps):
    f(); t = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - t) / reps
R = 3000
def obsFull(): fe.observeFloat32(b.F, b.I, b.PL, b.MO, al, n, out32, out64, *mode)
def obsNoLand():
    m = list(mode); m[-1] = False
    fe.observeFloat32(b.F, b.I, b.PL, b.MO, al, n, out32, out64, *m)
def copyOnly():
    out64[:n] = 0.0
    for a in range(n): pass
def slotsOnly():
    ids = np.empty(5, np.int64)
    for a in range(n): fe.slotIds(b.F, b.I, b.PL, int(al[a]), ids, 3, 2)
tObs = timeit(obsFull, R); tNoLand = timeit(obsNoLand, R)
print(f"observe (float32, {n} games): {tObs*1e6:.1f} us = {tObs/n*1e9:.0f} ns/game;  without landing: {tNoLand*1e6:.1f} us -> landing {(tObs-tNoLand)*1e6:.1f} us ({(tObs-tNoLand)/tObs*100:.0f}% of observe)")
# step cost: restore state each rep (copy cost measured separately)
def restore():
    b.F[:] = F0; b.I[:] = I0; b.PL[:] = PL0; b.MO[:] = MO0; b.BU[:] = BU0; b.MT[:] = MT0
def stepFull():
    restore(); b.step(al, np.full(n, 0.5), np.zeros(n), 10)
tCopy = timeit(restore, 300)
tStep = timeit(stepFull, 300) - tCopy
print(f"step 10 frames of {n} games: {tStep*1e6:.1f} us = {tStep/n/10*1e9:.0f} ns per game-frame (restore cost {tCopy*1e6:.0f} us subtracted)")
fused = timeit(lambda: (restore(), b.stepObserve(al.copy(), n, y, applied, smooth, 10))[1], 300) - tCopy
print(f"fused stepObserve: {fused*1e6:.1f} us  (= step {tStep*1e6:.1f} + observe {tObs*1e6:.1f} = {(tStep+tObs)*1e6:.1f})")
