"""Duration of 200 ES children (one evaluation each), then list-scheduling simulation of the generation on W workers."""
import sys, os, time, heapq
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import proSprudler as ps
net = ps.loadNetwork("models/rep10L_2026-10-24_02-52-46.pt")
flat = ps.flattenNetwork(net)
kids, _ = ps.betterBreed(flat, 0.05, 200)
seeds = [int(x) for x in np.random.RandomState(1).randint(0, 2**31, 200)]
LAYOUT = (3, 2, 1, 2, 0, 1)
def ev(n): return ps.evaluateNetworkFast(n, seeds, 100_000, None, 0, 0.0, 1 - 0.5 ** 10, LAYOUT, 0.0, 1.0, 10)
ev(ps.unflattenNetwork(kids[0], net))
d = []
for k in kids:
    n = ps.unflattenNetwork(k, net)
    t = time.perf_counter(); ev(n); d.append(time.perf_counter() - t)
d = np.array(d)
print(f"child eval time: mean {d.mean()*1e3:.0f} ms, min {d.min()*1e3:.0f}, max {d.max()*1e3:.0f}, p90 {np.percentile(d,90)*1e3:.0f}, cv {d.std()/d.mean():.2f}")
def sim(order, W):
    h = [0.0] * W
    for x in order:
        t = heapq.heappop(h); heapq.heappush(h, t + x)
    return max(h)
rng = np.random.RandomState(0)
for W in (19, 31):
    ideal = d.sum() / W
    fifo = np.mean([sim(rng.permutation(d), W) for _ in range(300)])
    lpt = sim(np.sort(d)[::-1], W)
    print(f"W={W}: ideal {ideal*1e3:.0f} ms, FIFO {fifo*1e3:.0f} ms (+{(fifo/ideal-1)*100:.1f}% idle tail), perfect longest-first {lpt*1e3:.0f} ms (+{(lpt/ideal-1)*100:.1f}%)")
    for halves in (2, 4):
        dd = np.repeat(d / halves, halves)  # each child's games split into 'halves' equal tasks (ignores the extra per-task overhead)
        f2 = np.mean([sim(rng.permutation(dd), W) for _ in range(300)])
        print(f"   split each child into {halves} tasks: {f2*1e3:.0f} ms (+{(f2/ideal-1)*100:.1f}%)")
