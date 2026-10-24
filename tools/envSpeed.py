import sys, os, time, importlib
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.expanduser(sys.argv[1]))
import numpy as np, proSprudler as ps
mods = {"orig": importlib.import_module("fastEnv_orig"), "new": importlib.import_module("fastEnv")}
cases = [("models/rep10L_2026-10-24_02-52-46.pt", 10, None, 0, 0.0), ("models/rep5C_2026-10-23_08-25-17.pt", 5, None, 0, 0.0), ("models/rep10L_2026-10-24_02-52-46.pt", 10, 30000, 20000, 0.5)]
pc = time.perf_counter
for model, rep, mx, mn, z in cases:
    net = ps.loadNetwork(os.path.join(os.path.expanduser(sys.argv[1]), model))
    layers = [(w.detach().numpy().astype(np.float32).T.copy(), b.detach().numpy().astype(np.float32)) for w, b in net.layers]
    smooth = 1 - 0.5 ** rep
    res = {k: [0.0, 0.0] for k in mods}
    for rnd in range(6):
        seeds = [int(x) for x in np.random.RandomState(rnd).randint(0, 2**31, 200)]
        for name in (("orig", "new") if rnd % 2 == 0 else ("new", "orig")):
            fe = mods[name]
            b = fe.FastBatch(seeds, mx, mn, z, 0.0, 1.0, stable=(3,2,1,2,0,1))
            alive, obs = b.startLoop(33); n = len(alive); ap = np.full(200, 0.5); fr = 0; te = 0.0
            t0 = pc()
            while n and fr < 100000:
                x = obs[:n]
                for L, (wT, bb) in enumerate(layers):
                    x = x @ wT + bb
                    x = np.maximum(x, 0) if L < len(layers) - 1 else 1 / (1 + np.exp(-np.clip(x, -50, 50)))
                t1 = pc(); n = b.stepObserve(alive, n, x, ap, smooth, rep); te += pc() - t1; fr += rep
            res[name][0] += pc() - t0; res[name][1] += te
    o, nw = res["orig"], res["new"]
    print(f"{os.path.basename(model)} repeat {rep} start {mn}-{mx}: whole eval orig {o[0]/6*1e3:.0f} ms -> new {nw[0]/6*1e3:.0f} ms ({(o[0]/nw[0]-1)*100:+.1f}% faster);  env part orig {o[1]/6*1e3:.0f} -> new {nw[1]/6*1e3:.0f} ms ({(o[1]/nw[1]-1)*100:+.1f}%)")
