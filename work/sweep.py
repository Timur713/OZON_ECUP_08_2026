import numpy as np, gc, sys, time, itertools, json
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats3 import build, targets
import lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    return rmsle(z,A@c)
VAL=378
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 12
TR=[t for t in range(186,VAL-29,STRIDE)]
print(f"anchors={len(TR)} stride={STRIDE}",flush=True)
X,names=build(TR); y=targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
Xv,_=build([VAL]); zv=np.log1p(targets([VAL]))
print("built",X.shape,flush=True)
CK=list(range(50,701,50))
GRID=[
 dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,lambda_l2=10.0),
 dict(learning_rate=0.05,num_leaves=255,min_data_in_leaf=100,feature_fraction=0.5,lambda_l2=20.0),
 dict(learning_rate=0.05,num_leaves=63, min_data_in_leaf=500,feature_fraction=0.7,lambda_l2=5.0),
 dict(learning_rate=0.03,num_leaves=511,min_data_in_leaf=300,feature_fraction=0.4,lambda_l2=50.0),
 dict(learning_rate=0.08,num_leaves=127,min_data_in_leaf=1000,feature_fraction=0.5,lambda_l2=30.0),
]
common=dict(bagging_fraction=0.8,bagging_freq=1,num_threads=8,verbose=-1,max_bin=255)
res=[]
for gi,g in enumerate(GRID):
    t0=time.time(); P={**common,**g}
    md=lgb.train({**P,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=700)
    mc=lgb.train({**P,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=700)
    mr=lgb.train({**P,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=700)
    PD={k:md.predict(Xv,num_iteration=k) for k in CK}
    PH={k:mc.predict(Xv,num_iteration=k)*mr.predict(Xv,num_iteration=k) for k in CK}
    best=None
    for kd in CK:
        for kh in CK:
            for w in (0.0,0.2,0.3,0.4,0.5):
                r=cal(w*PD[kd]+(1-w)*PH[kh],zv)
                if best is None or r<best[0]: best=(r,kd,kh,w)
    res.append((best[0],gi,g,best[1],best[2],best[3]))
    print(f"cfg{gi} {g} -> cal={best[0]:.5f} kd={best[1]} kh={best[2]} w={best[3]} ({time.time()-t0:.0f}s)",flush=True)
    np.save(W+f"work/sweep_pred_{gi}.npy",best[3]*PD[best[1]]+(1-best[3])*PH[best[2]])
res.sort()
print("\nRANKING:")
for r in res: print(f"  {r[0]:.5f}  cfg{r[1]} kd={r[3]} kh={r[4]} w={r[5]}  {r[2]}")
json.dump([[r[0],r[1],r[2],r[3],r[4],r[5]] for r in res],open(W+'work/sweep_res.json','w'),indent=1)
print("DONE",flush=True)
