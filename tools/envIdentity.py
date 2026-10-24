"""orig vs optimized fastEnv: same trained nets, lockstep, every decision step all observations + alive sets must be bit-identical."""
import sys, os, time, importlib
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.expanduser(sys.argv[1]))   # argv[1]: tensorNetwork dir (models, proSprudler)
import numpy as np
import proSprudler as ps
new = importlib.import_module("fastEnv"); old = importlib.import_module("fastEnv_orig")
CASES = [  # model, repeat, layout, maxStart, minStart, zero, monsterFraction, monsterMult, games
    ("models/rep10L_2026-10-24_02-52-46.pt", 10, (3,2,1,2,0,1), None, 0, 0.0, 0.0, 1.0, 200),
    ("models/rep10L_2026-10-24_02-52-46.pt", 10, (3,2,1,2,0,1), 30000, 20000, 0.5, 0.3, 2.0, 200),
    ("models/rep5C_2026-10-23_08-25-17.pt", 5, (3,2,1,2,0,1), 30000, 0, 0.0, 0.0, 1.0, 200),
    ("models/rep6A_2026-10-23_09-31-58.pt", 6, (3,2,1,2,0,1), None, 0, 0.0, 0.5, 2.0, 200),
]
tot = 0
for model, rep, lay, mx, mn, z, mf, mm, G in CASES:
    net = ps.loadNetwork(os.path.join(os.path.expanduser(sys.argv[1]), model))
    layers = [(w.detach().numpy().astype(np.float32).T.copy(), b.detach().numpy().astype(np.float32)) for w, b in net.layers]
    seeds = [int(x) for x in np.random.RandomState(7).randint(0, 2**31, G)]
    smooth = 1 - 0.5 ** rep
    bo = old.FastBatch(seeds, mx, mn, z, mf, mm, stable=lay); bn = new.FastBatch(seeds, mx, mn, z, mf, mm, stable=lay)
    ao, oo = bo.startLoop(33); an, on = bn.startLoop(33); n = len(ao); apo = np.full(G, 0.5); apn = np.full(G, 0.5)
    assert np.array_equal(oo, on)
    fr = 0
    while n and fr < 100000:
        x = oo[:n]
        for L, (wT, bb) in enumerate(layers):
            x = x @ wT + bb
            x = np.maximum(x, 0) if L < len(layers) - 1 else 1 / (1 + np.exp(-np.clip(x, -50, 50)))
        no = bo.stepObserve(ao, n, x, apo, smooth, rep); nn = bn.stepObserve(an, n, x, apn, smooth, rep)
        assert no == nn and np.array_equal(ao[:no], an[:nn]), f"alive differs at frame {fr}"
        assert np.array_equal(oo[:no], on[:nn]), f"observation differs at frame {fr}"
        n = no; fr += rep; tot += n * rep
    assert np.array_equal(bo.scores(), bn.scores()) and np.array_equal(bo.causes(), bn.causes())
    print(f"{os.path.basename(model)} repeat {rep} start {mn}-{mx} monsters {mf}: {fr:,} frames, mean score {bo.scores().mean():,.0f}, max {bo.scores().max():,.0f} -> IDENTICAL")
print(f"IDENTICAL over ~{tot:,} game-frames")
