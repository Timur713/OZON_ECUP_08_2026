import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats2 import build, targets
import lightgbm as lgb
from datetime import date, timedelta
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
D0=date(2025,1,1); NU=250000
def ds(t): return (D0+timedelta(days=int(t))).isoformat()
def rmsle(z,zh): return float(np.sqrt(np.mean((z-np.clip(zh,0,None))**2)))
M1,M2=2.3232887,10.7633307     # measured on public LB

VAL=378
TR=[t for t in range(186,VAL-29,12)]      # targets end <= 2026-01-14
print("train anchors:",[ds(t) for t in TR],flush=True)
t0=time.time()
Xtr,names=build(TR,verbose=True); ytr=targets(TR)
print("Xtr",Xtr.shape,f"{Xtr.nbytes/1e9:.2f}GB",time.time()-t0,flush=True)
Xva,_=build([VAL]); yva=targets([VAL])
ztr=np.log1p(ytr); zva=np.log1p(yva); btr=(ytr>0).astype(np.int8)
print("features:",len(names),flush=True)

base=dict(learning_rate=0.03,num_leaves=255,min_data_in_leaf=100,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
def cal(p,z):
    A=np.vstack([p,np.ones_like(p)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    return rmsle(z,A@c),float(c[0]),float(c[1])

print("\n--- direct L2 ---",flush=True)
dtr=lgb.Dataset(Xtr,ztr,feature_name=names)
md=lgb.train({**base,'objective':'regression'},dtr,num_boost_round=3000,
             valid_sets=[lgb.Dataset(Xva,zva,feature_name=names)],
             callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(300)])
pd_=md.predict(Xva,num_iteration=md.best_iteration)
r,a,b=cal(pd_,zva); print(f"direct: iter={md.best_iteration} raw={rmsle(zva,pd_):.4f} cal={r:.4f} a={a:.3f} b={b:+.3f}",flush=True)
ND_=md.best_iteration

print("\n--- hurdle ---",flush=True)
mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(Xtr,btr,feature_name=names),num_boost_round=3000,
             valid_sets=[lgb.Dataset(Xva,(yva>0).astype(np.int8),feature_name=names)],
             callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(300)])
NC_=mc.best_iteration
pos=btr==1
mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr[pos],ztr[pos],feature_name=names),
             num_boost_round=3000,valid_sets=[lgb.Dataset(Xva[yva>0],zva[yva>0],feature_name=names)],
             callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(300)])
NR_=mr.best_iteration
pc=mc.predict(Xva,num_iteration=NC_); pr=mr.predict(Xva,num_iteration=NR_)
r,a,b=cal(pc*pr,zva); print(f"hurdle: iters={NC_}/{NR_} raw={rmsle(zva,pc*pr):.4f} cal={r:.4f} a={a:.3f} b={b:+.3f}",flush=True)

best=(9,None)
for w in np.arange(0,1.01,0.1):
    r,a,b=cal(w*pd_+(1-w)*(pc*pr),zva)
    if r<best[0]: best=(r,w,a,b)
print(f"\nBEST blend w_direct={best[1]:.1f} -> cal RMSLE={best[0]:.4f} (a={best[2]:.3f} b={best[3]:+.3f})",flush=True)
imp=sorted(zip(names,md.feature_importance('gain')),key=lambda x:-x[1])[:20]
print("\nTOP FEATURES:"); [print(f"  {n:24s} {g:,.0f}") for n,g in imp]
np.save(W+'work/v2_iters.npy',np.array([ND_,NC_,NR_,best[1],best[2]]))
np.save(W+'work/v2_val.npy',np.vstack([pd_,pc,pr,zva]))
print("\nDONE v2 validation",flush=True)
