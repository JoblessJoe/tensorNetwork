'''
Keeps everything a net has learned except WHEN it shoots: the shoot output's weights (row 1 of the last layer) become tiny random values and its bias 0,
so the shoot logit starts near 0 instead of saturated at about +25 (where ES noise can never flip a shot and a shot penalty gets no signal).
Writes a one-net pool file that can be passed as `pool=` of a nightRun phase (ES takes the first net of a pool as its parent).

    python resetShootHead.py models/x.pt models/x_headReset_pool.pt        (x.pt: a saved net, or a *_pool.pt: its first net is used)
'''
import sys
import torch
import proSprudler as ps

src, dst = sys.argv[1], sys.argv[2]
net = ps.loadPool(src)[0] if src.endswith("_pool.pt") else ps.loadNetwork(src)
w, b = net.layers[-1]
assert w.shape[0] == 2, "expected the 2-output network (steer, shoot)"
generator = torch.Generator().manual_seed(0)
with torch.no_grad():
    w[1] = torch.randn(w.shape[1], generator=generator) * 1e-3
    b[1] = 0.0
ps.savePool([net], dst)
print(f"saved {dst}: shoot row reset ({w.shape[1]} weights ~N(0, 0.001), bias 0)")
