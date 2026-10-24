import numpy as np, time
from numba import njit
try:
    import scipy; print("scipy", scipy.__version__)
except Exception as e: print("no scipy:", e)
rng = np.random.RandomState(0)
sizes = [33, 33, 33, 33, 2]
Ws = [rng.randn(sizes[i], sizes[i+1]).astype(np.float32) * 0.3 for i in range(4)]
Bs = [rng.randn(sizes[i+1]).astype(np.float32) * 0.1 for i in range(4)]
def npFwd(x):
    for L in range(4):
        x = x @ Ws[L] + Bs[L]
        if L < 3: x = np.maximum(x, 0)
        else:
            with np.errstate(over="ignore"): x = 1 / (1 + np.exp(-x))
    return x
@njit(cache=False)
def nbDot(x, W0, B0, W1, B1, W2, B2, W3, B3):       # BLAS matmuls inside compiled code, elementwise ops compiled
    h = np.dot(x, W0) + B0
    h = np.maximum(h, 0)
    h = np.dot(h, W1) + B1
    h = np.maximum(h, 0)
    h = np.dot(h, W2) + B2
    h = np.maximum(h, 0)
    o = np.dot(h, W3) + B3
    return o
def fused(x):
    o = nbDot(x, Ws[0], Bs[0], Ws[1], Bs[1], Ws[2], Bs[2], Ws[3], Bs[3])
    with np.errstate(over="ignore"): return 1 / (1 + np.exp(-o))
pc = time.perf_counter
for n in (5, 20, 50, 100, 200):
    x = rng.rand(n, 33).astype(np.float32)
    a = npFwd(x); b = fused(x); same = np.array_equal(a, b); pre = np.array_equal(a, 1/(1+np.exp(-nbDot(x, Ws[0], Bs[0], Ws[1], Bs[1], Ws[2], Bs[2], Ws[3], Bs[3]))))
    R = 4000
    t = pc(); [npFwd(x) for _ in range(R)]; t1 = (pc() - t) / R
    t = pc(); [fused(x) for _ in range(R)]; t2 = (pc() - t) / R
    t = pc(); [nbDot(x, Ws[0], Bs[0], Ws[1], Bs[1], Ws[2], Bs[2], Ws[3], Bs[3]) for _ in range(R)]; t3 = (pc() - t) / R
    print(f"n={n:3d}: numpy {t1*1e6:6.1f} us | numba-dot + numpy sigmoid {t2*1e6:6.1f} us | numba-dot only {t3*1e6:6.1f} us | outputs identical: {same}")
