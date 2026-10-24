"""Time share of decision steps by number of alive games (rep10L, 200 games, 5 different seed sets)."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np, proSprudler as ps
from fastEnv import FastBatch
LAYOUT = (3, 2, 1, 2, 0, 1)
net = ps.loadNetwork("models/rep10L_2026-10-24_02-52-46.pt")
layers = [(w.detach().numpy().astype(np.float32).T.copy(), b.detach().numpy().astype(np.float32)) for w, b in net.layers]
smooth = 1 - 0.5 ** 10; pc = time.perf_counter
edges = [0, 3, 10, 30, 100, 201]
tf = np.zeros(5); ts = np.zeros(5); cnt = np.zeros(5)
for rep in range(5):
    b = FastBatch([int(s) + rep * 977 for s in np.random.RandomState(rep).randint(0, 2**31, 200)], None, 0, 0.0, 0.0, 1.0, stable=LAYOUT)
    alive, obs = b.startLoop(33); n = len(alive); applied = np.full(200, 0.5); fr = 0
    while n and fr < 100000:
        k = np.searchsorted(edges, n, side="right") - 1
        t0 = pc(); x = obs[:n]
        for L, (wT, bb) in enumerate(layers):
            x = x @ wT + bb
            x = np.maximum(x, 0) if L < len(layers) - 1 else 1 / (1 + np.exp(-np.clip(x, -50, 50)))
        t1 = pc(); n = b.stepObserve(alive, n, x, applied, smooth, 10); t2 = pc()
        tf[k] += t1 - t0; ts[k] += t2 - t1; cnt[k] += 1; fr += 10
tot = (tf + ts).sum()
print("alive games | steps | forward share | env share  (of total evaluation time)")
for k in range(5):
    print(f"{edges[k]:3d}-{edges[k+1]-1:3d}     | {int(cnt[k]):6d} | {tf[k]/tot*100:5.1f}%        | {ts[k]/tot*100:5.1f}%     (fwd {tf[k]/max(cnt[k],1)*1e6:.1f} us/step, env {ts[k]/max(cnt[k],1)*1e6:.1f} us/step)")
print(f"total forward {tf.sum()/tot*100:.0f}%, env {ts.sum()/tot*100:.0f}%")
