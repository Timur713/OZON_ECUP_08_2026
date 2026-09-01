"""Evaluate every trained sequence model: quality, correlation, stack contribution."""
import numpy as np, sys, glob, os
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats2
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1,M2=2.3232887,10.7633307
zv=np.log1p(feats2.targets([378]))
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(x):
    A=np.vstack([x,np.ones_like(x)]).T; c,_,_,_=np.linalg.lstsq(A,zv,rcond=None); return rmsle(zv,A@c)
P={'v4':np.load(W+'work/v4_262_valpred.npy')}
for f in sorted(glob.glob(W+'work/*_val.npy')):
    n=os.path.basename(f)[:-8]
    if n in ('seq','tcn365','zva378','zva','cb'): pass
    a=np.load(f).astype(np.float64)
    if a.shape==zv.shape and n not in P and 'zva' not in n: P[n]=a
print(f"{'model':14s} {'val cal':>9}  corr with v4 / tcn365")
t3=P.get('tcn365')
for k,v in P.items():
    c1=np.corrcoef(v,P['v4'])[0,1]; c2=np.corrcoef(v,t3)[0,1] if t3 is not None else float('nan')
    print(f"{k:14s} {cal(v):9.5f}  {c1:.5f} {c2:.5f}")
def stack(keys):
    X=np.vstack([P[k] for k in keys]+[np.ones_like(zv)]).T
    c,_,_,_=np.linalg.lstsq(X,zv,rcond=None); return rmsle(zv,X@c),c
base=['v4','seq','tcn365'] if 'seq' in P else ['v4','tcn365']
base=[b for b in base if b in P]
r0,_=stack(base); print(f"\nbaseline stack {base}: {r0:.5f}")
for k in P:
    if k in base: continue
    r,_=stack(base+[k]); print(f"  + {k:14s} -> {r:.5f}   gain {r0-r:+.5f}")
