import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats2 import build, targets
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    return rmsle(z,A@c),float(c[0]),float(c[1])
VAL=378; TR=[t for t in range(186,VAL-29,12)]
t0=time.time()
X,names=build(TR); y=targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8)
Xv,_=build([VAL]); yv=targets([VAL]); zv=np.log1p(yv)
print("built",X.shape,time.time()-t0,flush=True)
CK=list(range(100,1501,100))
P=dict(depth=8,learning_rate=0.05,l2_leaf_reg=6.0,border_count=128,
       thread_count=8,verbose=200,random_seed=1,allow_writing_files=False)
res={}
print("\n--- CatBoost direct ---",flush=True)
md=CatBoostRegressor(iterations=1500,loss_function='RMSE',**P)
md.fit(Pool(X,z))
pdv=np.vstack([md.predict(Xv,ntree_end=k) for k in CK])
print("\n--- CatBoost binary ---",flush=True)
mc=CatBoostClassifier(iterations=1500,loss_function='Logloss',**P)
mc.fit(Pool(X,b))
pcv=np.vstack([mc.predict_proba(Xv,ntree_end=k)[:,1] for k in CK])
print("\n--- CatBoost cond-value ---",flush=True)
pos=b==1
mr=CatBoostRegressor(iterations=1500,loss_function='RMSE',**P)
mr.fit(Pool(X[pos],z[pos]))
prv=np.vstack([mr.predict(Xv,ntree_end=k) for k in CK])
del X; gc.collect()
print(f"\n{'iter':>6} {'direct_cal':>11} {'hurdle_cal':>11}",flush=True)
bd=(9,0);bh=(9,0)
for i,k in enumerate(CK):
    rd,_,_=cal(pdv[i],zv); rh,_,_=cal(pcv[i]*prv[i],zv)
    if rd<bd[0]: bd=(rd,i)
    if rh<bh[0]: bh=(rh,i)
    print(f"{k:6d} {rd:11.5f} {rh:11.5f}",flush=True)
print(f"\nCB best direct iter={CK[bd[1]]} cal={bd[0]:.5f} | best hurdle iter={CK[bh[1]]} cal={bh[0]:.5f}",flush=True)
cb=None
for w in np.arange(0,1.01,0.1):
    r,_,_=cal(w*pdv[bd[1]]+(1-w)*(pcv[bh[1]]*prv[bh[1]]),zv)
    if cb is None or r<cb[0]: cb=(r,w)
print(f"CB best blend w={cb[1]:.1f} cal={cb[0]:.5f}",flush=True)
cbz=cb[1]*pdv[bd[1]]+(1-cb[1])*(pcv[bh[1]]*prv[bh[1]])
np.save(W+'work/cb_val.npy',np.vstack([cbz,zv]))
# combine with the LightGBM validation blend
lp,lc,lr_,_=np.load(W+'work/v2_val.npy')
lgbz=0.5*lp+0.5*(lc*lr_)
print(f"\nLGBM val cal = {cal(lgbz,zv)[0]:.5f}",flush=True)
best=None
for w in np.arange(0,1.01,0.05):
    r,_,_=cal(w*lgbz+(1-w)*cbz,zv)
    if best is None or r<best[0]: best=(r,w)
print(f"LGBM+CB blend: w_lgbm={best[1]:.2f} cal={best[0]:.5f}",flush=True)
np.save(W+'work/cb_cfg.npy',np.array([CK[bd[1]],CK[bh[1]],cb[1],best[1],best[0]]))
print("DONE",flush=True)
