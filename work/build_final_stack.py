"""Retrain the best GBDT config on TR+VAL, predict at t=408, stack with both TCNs."""
import numpy as np, gc, sys, csv, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats3 import build, targets
import lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1,M2=2.3232887,10.7633307
VAL=378; FINAL=408
CFG=dict(learning_rate=0.03,num_leaves=511,min_data_in_leaf=300,feature_fraction=0.4,
         lambda_l2=50.0,bagging_fraction=0.8,bagging_freq=1,num_threads=8,verbose=-1,max_bin=255)
KD,KH,WB=300,300,0.3
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
TR=[t for t in range(186,VAL-29,12)]
# --- 1. validation-fitted stack weights (cfg3 preds already saved by sweep)
zv=np.log1p(targets([VAL]))
cf3=np.load(W+'work/sweep_pred_3.npy')
t1=np.load(W+'work/seq_val.npy').astype(np.float64)
t3=np.load(W+'work/tcn365_val.npy').astype(np.float64)
X=np.vstack([cf3,t1,t3,np.ones_like(zv)]).T
c,_,_,_=np.linalg.lstsq(X,zv,rcond=None)
print(f"stack val RMSLE={rmsle(zv,X@c):.5f}  weights={[round(float(x),4) for x in c]}",flush=True)
np.save(W+'work/final_stack_w.npy',c)
# --- 2. retrain cfg3 on TR+VAL, predict at FINAL
TR2=TR+[VAL]; sc=len(TR2)/len(TR)
kd,kh=int(KD*sc),int(KH*sc)
t0=time.time()
Xt,names=build(TR2); y=targets(TR2); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
print("built",Xt.shape,time.time()-t0,flush=True)
Xp,_=build([FINAL])
PD=[];PH=[]
for seed in (1,2):
    P={**CFG,'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed}
    md=lgb.train({**P,'objective':'regression'},lgb.Dataset(Xt,z,feature_name=names),num_boost_round=kd)
    mc=lgb.train({**P,'objective':'binary'},lgb.Dataset(Xt,b,feature_name=names),num_boost_round=kh)
    mr=lgb.train({**P,'objective':'regression'},lgb.Dataset(Xt[pos],z[pos],feature_name=names),num_boost_round=kh)
    PD.append(md.predict(Xp)); PH.append(mc.predict(Xp)*mr.predict(Xp))
    print(f"  seed{seed} {time.time()-t0:.0f}s",flush=True)
del Xt; gc.collect()
cf3_fin=WB*np.mean(PD,0)+(1-WB)*np.mean(PH,0)
np.save(W+'work/cfg3_final.npy',cf3_fin)
# --- 3. scale-match TCN finals to their validation scale, apply stack weights
f1=np.load(W+'work/seq_final.npy').astype(np.float64)
f3=np.load(W+'work/tcn365_final.npy').astype(np.float64)
def match(fin,val,ref): return (fin-val.mean())/val.std()*ref.std()+ref.mean()
f1s=match(f1,t1,cf3); f3s=match(f3,t3,cf3)
zh=c[0]*cf3_fin+c[1]*f1s+c[2]*f3s+c[3]
np.save(W+'work/v7_zh_final.npy',zh)
sl=1.08
v=np.clip(sl*zh+(M1-sl*zh.mean()),0,None); np.save(W+'work/v7_v.npy',v)
p=np.expm1(v); uids=np.load(W+'work/mat/uids.npy')
with open(W+'submissions/20_v7_final_stack.csv','w',newline='') as f:
    cw=csv.writer(f); cw.writerow(['user_id','predict'])
    for u,x in zip(uids,p): cw.writerow([int(u),float(x)])
print(f"wrote 20_v7_final_stack.csv mean_log1p={np.log1p(p).mean():.6f} sd={v.std():.4f}",flush=True)
print("DONE",flush=True)
