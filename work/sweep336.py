"""Re-run GBDT config selection on the CORRECTED (rising-season) validation anchor.
Every previous config choice was made on anchor 378, which we proved ranks inverted."""
import numpy as np, gc, sys, time, json
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats3, lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
VAL=336                       # target 2025-12-03..2026-01-01, seasonality +4.55%
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None); return rmsle(z,A@c)
TR=[t for t in range(120,VAL-29,12)]
print(f"anchors={len(TR)} val={VAL}",flush=True)
X,names=feats3.build(TR); y=feats3.targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
Xv,_=feats3.build([VAL]); zv=np.log1p(feats3.targets([VAL]))
print(f"X={X.shape} {X.nbytes/1e9:.2f}GB",flush=True)
common=dict(bagging_fraction=0.8,bagging_freq=1,num_threads=8,verbose=-1,max_bin=255)
GRID=[
 ("A base",   dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,lambda_l2=10.0)),
 ("B deep",   dict(learning_rate=0.03,num_leaves=511,min_data_in_leaf=300,feature_fraction=0.4,lambda_l2=50.0)),
 ("C shallow",dict(learning_rate=0.05,num_leaves=31, min_data_in_leaf=1000,feature_fraction=0.8,lambda_l2=5.0)),
 ("D strongreg",dict(learning_rate=0.04,num_leaves=255,min_data_in_leaf=2000,feature_fraction=0.35,lambda_l2=200.0)),
 ("E lowlr",  dict(learning_rate=0.02,num_leaves=127,min_data_in_leaf=500,feature_fraction=0.5,lambda_l2=20.0)),
]
CK=list(range(50,901,50)); out={}
for tag,g in GRID:
    t0=time.time(); P={**common,**g}
    md=lgb.train({**P,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=900)
    mc=lgb.train({**P,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=900)
    mr=lgb.train({**P,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=900)
    PD={k:md.predict(Xv,num_iteration=k) for k in CK}
    PH={k:mc.predict(Xv,num_iteration=k)*mr.predict(Xv,num_iteration=k) for k in CK}
    best=None
    for kd in CK:
        for kh in CK:
            for w in (0.0,0.2,0.3,0.4,0.5):
                r=cal(w*PD[kd]+(1-w)*PH[kh],zv)
                if best is None or r<best[0]: best=(r,kd,kh,w)
    out[tag]=best; print(f"{tag:11s} cal={best[0]:.5f} kd={best[1]} kh={best[2]} w={best[3]} ({time.time()-t0:.0f}s)",flush=True)
    del PD,PH; gc.collect()
json.dump({k:[float(v[0]),v[1],v[2],v[3]] for k,v in out.items()},open(W+'work/sweep336.json','w'),indent=1)
print("\nRANKING (on the CORRECTED anchor):")
for tag,b in sorted(out.items(),key=lambda x:x[1][0]): print(f"  {b[0]:.5f}  {tag}")
print("DONE",flush=True)
