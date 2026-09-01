import numpy as np, gc, sys
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats import build, targets
import lightgbm as lgb
from scipy.optimize import brentq
from datetime import date, timedelta
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
D0=date(2025,1,1); NU=250000; FINAL=408
def ds(t): return (D0+timedelta(days=int(t))).isoformat()
def rmsle(z,zh): return float(np.sqrt(np.mean((z-np.clip(zh,0,None))**2)))
def shift_to(p,target):
    p=np.clip(p,1e-9,1-1e-9); lg=np.log(p/(1-p))
    d=brentq(lambda d:(1/(1+np.exp(-(lg+d)))).mean()-target,-8,8)
    return 1/(1+np.exp(-(lg+d))),d
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.7,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,num_threads=8,verbose=-1,max_bin=127)
NR=600
def fit_predict(TR,PRED):
    Xtr,names=build(TR,verbose=False); ytr=targets(TR)
    ztr=np.log1p(ytr); btr=(ytr>0).astype(np.int8); pos=btr==1
    mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(Xtr,btr,feature_name=names),num_boost_round=NR)
    mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr[pos],ztr[pos],feature_name=names),num_boost_round=NR)
    md=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr,ztr,feature_name=names),num_boost_round=NR)
    del Xtr; gc.collect()
    Xp,_=build([PRED],verbose=False)
    out=(mc.predict(Xp),mr.predict(Xp),md.predict(Xp))
    del Xp; gc.collect(); return out

# ---- 1. rolling estimate of the post-p-shift affine slope ----
print("=== rolling slope estimation ===",flush=True)
slopes=[]
for V in [336,357,378]:
    TR=[t for t in range(210,V-29,17)]
    pc,pr,pd=fit_predict(TR,V)
    yv=targets([V]); zv=np.log1p(yv); Ptrue=(yv>0).mean()
    pcal,_=shift_to(pc,Ptrue)
    zh=0.6*(pcal*pr)+0.4*pd
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,zv,rcond=None)
    slopes.append(float(c[0]))
    print(f"  val {ds(V)}: a={c[0]:.4f} b={c[1]:+.4f} rmsle={rmsle(zv,A@c):.4f} "
          f"(uncal {rmsle(zv,zh):.4f})",flush=True)
A_SLOPE=float(np.median(slopes)); print(f"  -> slope used: {A_SLOPE:.4f}",flush=True)

# ---- 2. final model ----
print("\n=== final model: train through 2026-01-14, predict 2026-02-13 ===",flush=True)
TR=[t for t in range(210,379,17)]
print("  anchors:",[ds(t) for t in TR],flush=True)
pc,pr,pd=fit_predict(TR,FINAL)
np.save(W+'work/final_pc.npy',pc); np.save(W+'work/final_pr.npy',pr); np.save(W+'work/final_pd.npy',pd)
print(f"  raw: mean p={pc.mean():.4f}  mean mu={pr.mean():.4f}  mean direct={pd.mean():.4f}",flush=True)

P_TARGET=0.5855; L_TARGET=2.4660
pcal,delta=shift_to(pc,P_TARGET)
zh=0.6*(pcal*pr)+0.4*pd
print(f"  after logit-shift (delta={delta:+.4f}): mean zhat={zh.mean():.4f}",flush=True)
b=L_TARGET-A_SLOPE*zh.mean()
zfin=np.clip(A_SLOPE*zh+b,0,None)
print(f"  affine a={A_SLOPE:.4f} b={b:+.4f} -> mean zfin={zfin.mean():.4f} (target L={L_TARGET})",flush=True)
uids=np.load(W+'work/mat/uids.npy')
def write(name,pred):
    import csv
    with open(W+'submissions/'+name,'w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['user_id','predict'])
        for u,p in zip(uids,pred): wr.writerow([int(u),float(p)])
    print("  wrote",name,flush=True)
import os; os.makedirs(W+'submissions',exist_ok=True)
write('01_probe_zeros.csv',np.zeros(NU))
write('02_probe_const100.csv',np.full(NU,100.0))
write('03_model_L2466.csv',np.expm1(zfin))
b2=2.4335-A_SLOPE*zh.mean(); write('04_model_L2434.csv',np.expm1(np.clip(A_SLOPE*zh+b2,0,None)))
np.save(W+'work/final_zh.npy',zh); np.save(W+'work/final_slope.npy',np.array([A_SLOPE]))
print("\nDONE",flush=True)
