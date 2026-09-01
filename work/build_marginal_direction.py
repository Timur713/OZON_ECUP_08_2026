#!/usr/bin/env python
"""P(buy) at the final anchor and the marginal-buyer direction p*(1-p).

Season shifts P(y>0) by x1.097 (FINDINGS 5).  A shift that converts buyers acts
hardest on users near the decision margin, so its signature is p*(1-p), which
is NOT a function of zhat and is nonlinear in p, hence absent from every base
in the pool.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/ecup/work")
import feats4, lightgbm as lgb

W = "/home/ubuntu/ecup/work/"
FINAL = 408
X, names = feats4.build([FINAL])
print("features", X.shape, flush=True)
ps = []
for s in (1, 2):
    m = lgb.Booster(model_file=f"{W}gbdtctl_s{s}_class.txt")
    ps.append(m.predict(X))
    print("classifier", s, "done", flush=True)
p = np.mean(ps, 0)
np.save(W + "pbuy408.npy", p.astype(np.float32))
marg = p * (1.0 - p)
np.save(W + "marginal408.npy", marg.astype(np.float32))
print(f"p: mean={p.mean():.5f} sd={p.std():.5f} min={p.min():.5f} max={p.max():.5f}")
print(f"p(1-p): mean={marg.mean():.5f} sd={marg.std():.5f}")
print(f"corr(p, p(1-p)) = {np.corrcoef(p, marg)[0,1]:.5f}")
