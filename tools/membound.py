import sys,os,time
sys.path.insert(0,os.getcwd())
import numpy as np, multiprocessing as mp
import proSprudler as ps
def ctrl(_):   # pure compute, tiny working set (~100 KB)
    x0=np.random.rand(200,33).astype(np.float32); w=np.random.rand(33,33).astype(np.float32)
    t=time.perf_counter()
    for _ in range(40000):
        x=x0
        for _ in range(3): x=np.maximum(x@w,0)
    return time.perf_counter()-t
def game(args):
    G,K=args
    net=ps.loadNetwork("models/rep10L_2026-10-24_02-52-46.pt"); seeds=list(range(1000,1000+G))
    f=lambda: ps.evaluateNetworkFast(net,seeds,100000,None,0,0.0,1-0.5**10,(3,2,1,2,0,1),0.0,1.0,10)
    f(); t=time.perf_counter()
    for _ in range(K): f()
    return (time.perf_counter()-t)/K/G*1e6   # us per game
def run(fn,arg,W):
    with mp.get_context("fork").Pool(W) as p: return float(np.mean(p.map(fn,[arg]*W)))
if __name__=="__main__":
    c1=run(ctrl,0,1); c16=run(ctrl,0,16); c31=run(ctrl,0,31)
    print(f"control (tiny working set, pure compute): W=16 per-worker speed {c1/c16*100:.0f}%, W=31 {c1/c31*100:.0f}% of W=1")
    for G,K in ((25,300),(100,100),(200,60),(800,20)):
        a=run(game,(G,K),1); b=run(game,(G,K),16); c=run(game,(G,K),31)
        print(f"games/eval {G:4d}: W=1 {a:6.1f} us/game, W=16 {b:6.1f} ({a/b*100:.0f}%), W=31 {c:6.1f} ({a/c*100:.0f}%)")
