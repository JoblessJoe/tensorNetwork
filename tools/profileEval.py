"""Profile of one network evaluation (as in ES training) + worker scaling. Run from the tensorNetwork dir: python profileEval.py"""
import sys, time, os
sys.path.insert(0, os.getcwd())
import numpy as np
import multiprocessing as mp
import proSprudler as ps
from fastEnv import FastBatch

LAYOUT = (3, 2, 1, 2, 0, 1)
SEEDS = list(range(1000, 1200))


def load(path):
    net = ps.loadNetwork(path)
    layers = [(w.detach().cpu().numpy().astype(np.float32).T.copy(), b.detach().cpu().numpy().astype(np.float32)) for w, b in net.layers]
    return net, layers


def evalOnce(net, repeat, smooth):
    return ps.evaluateNetworkFast(net, SEEDS, 100_000, None, 0, 0.0, smooth, LAYOUT, 0.0, 1.0, repeat)


def breakdown(path, repeat, smooth):
    net, layers = load(path)
    nIn = layers[0][0].shape[0]
    evalOnce(net, repeat, smooth)  # warm-up (numba)
    pc = time.perf_counter
    batch = FastBatch(SEEDS, None, 0, 0.0, 0.0, 1.0, stable=LAYOUT, difficulty=False)
    alive, obs = batch.startLoop(nIn)
    n = len(alive)
    applied = np.full(len(SEEDS), 0.5)
    tFwd = tStep = 0.0
    steps = 0
    sumN = 0
    frames = 0
    tAll = pc()
    while n and frames < 100_000:
        t0 = pc()
        x = obs[:n]
        for L, (wT, b) in enumerate(layers):
            x = x @ wT + b
            if L < len(layers) - 1:
                x = np.maximum(x, 0)
            else:
                with np.errstate(over="ignore"):
                    x = 1 / (1 + np.exp(-x))
        t1 = pc()
        n = batch.stepObserve(alive, n, x, applied, smooth, repeat, decide=False)
        t2 = pc()
        tFwd += t1 - t0
        tStep += t2 - t1
        steps += 1
        sumN += n
        frames += repeat
    tot = pc() - tAll
    gameFrames = sumN * repeat  # approx: alive games after each decision * frames per decision
    print(f"  {os.path.basename(path)} repeat {repeat}: total {tot:.2f}s, {steps} decision steps, mean alive games {sumN / steps:.0f}")
    print(f"    forward (numpy) {tFwd / tot * 100:.0f}%  = {tFwd / steps * 1e6:.1f} us/step;  stepObserve (env) {tStep / tot * 100:.0f}%  = {tStep / steps * 1e6:.1f} us/step;  other {100 - (tFwd + tStep) / tot * 100:.0f}%")
    print(f"    env cost per game-frame ~ {tStep / max(gameFrames, 1) * 1e9:.0f} ns;  per decision per alive game: fwd {tFwd / sumN * 1e9:.0f} ns, env {tStep / sumN * 1e9:.0f} ns")


def numpyOverhead(layers):
    """cost of the forward pass by batch size: fixed call overhead vs per-row compute"""
    pc = time.perf_counter
    for n in (1, 10, 50, 200, 1000, 5000):
        x0 = np.random.rand(n, layers[0][0].shape[0]).astype(np.float32)
        t = pc()
        for _ in range(2000):
            x = x0
            for L, (wT, b) in enumerate(layers):
                x = x @ wT + b
                if L < len(layers) - 1:
                    x = np.maximum(x, 0)
                else:
                    x = 1 / (1 + np.exp(-x))
        dt = (pc() - t) / 2000
        print(f"    batch {n:5d}: {dt * 1e6:7.1f} us per forward  ({dt / n * 1e9:8.1f} ns per row)")


def worker(args):
    path, repeat, smooth, K = args
    net, _ = load(path)
    evalOnce(net, repeat, smooth)
    t = time.perf_counter()
    for _ in range(K):
        evalOnce(net, repeat, smooth)
    return time.perf_counter() - t


def scaling(path, repeat, smooth, K=60):
    ctx = mp.get_context("fork")
    base = None
    for W in (1, 2, 4, 8, 12, 16, 20, 24, 28, 31):
        with ctx.Pool(W) as p:
            ts = p.map(worker, [(path, repeat, smooth, K)] * W)
        per = np.mean(ts) / K
        base = base or per
        print(f"    W={W:2d}: {per:5.2f} s per eval per worker (slowest {max(ts) / K:.2f})  -> per-worker speed {base / per * 100:5.1f}% of W=1, total throughput x{W * base / per:5.1f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    m10 = ("models/rep10L_2026-10-24_02-52-46.pt", 10, 1 - 0.5 ** 10)
    m5 = ("models/rep5C_2026-10-23_08-25-17.pt", 5, 1 - 0.5 ** 5)
    if which in ("all", "breakdown"):
        print("== time breakdown, one process")
        for p, r, s in (m5, m10):
            breakdown(p, r, s)
        print("== numpy forward cost by batch size (33-33-33-33-2 net)")
        numpyOverhead(load(m5[0])[1])
    if which in ("all", "scaling"):
        print("== worker scaling (same job on every worker), rep10L")
        scaling(*m10)
