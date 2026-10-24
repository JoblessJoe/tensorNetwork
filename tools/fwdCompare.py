"""evaluateNetworkFast: numpy forward vs compiled forward - identical scores? and time (alternating rounds, one process, idle machine)"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np, proSprudler as ps
LAY = (3, 2, 1, 2, 0, 1)
print("COMPILED_FORWARD available:", ps.COMPILED_FORWARD)
cases = [("models/rep10L_2026-10-24_02-52-46.pt", 10, None, 0, 0.0, 0.0), ("models/rep5C_2026-10-23_08-25-17.pt", 5, None, 0, 0.0, 0.0),
         ("models/rep6A_2026-10-23_09-31-58.pt", 6, 30000, 20000, 0.5, 0.3), ("models/rep10B_2026-10-23_15-12-03.pt", 10, None, 0, 0.0, 0.0)]
for model, rep, mx, mn, z, mf in cases:
    net = ps.loadNetwork(model); sm = 1 - 0.5 ** rep
    def ev(seeds, flag):
        ps.COMPILED_FORWARD = flag
        return ps.evaluateNetworkFast(net, seeds, 100_000, mx, mn, z, sm, LAY, mf, 2.0 if mf else 1.0, rep)
    ev([1, 2, 3], True); ev([1, 2, 3], False)
    tt = {True: 0.0, False: 0.0}; same = True
    for r in range(6):
        seeds = [int(x) for x in np.random.RandomState(r).randint(0, 2**31, 200)]
        res = {}
        for flag in ((True, False) if r % 2 == 0 else (False, True)):
            t = time.perf_counter(); res[flag] = ev(seeds, flag); tt[flag] += time.perf_counter() - t
        same &= res[True] == res[False]
    print(f"{os.path.basename(model)} repeat {rep}: scores identical: {same};  numpy forward {tt[False]/6*1e3:.0f} ms -> compiled {tt[True]/6*1e3:.0f} ms  ({(tt[False]/tt[True]-1)*100:+.1f}% faster)")
