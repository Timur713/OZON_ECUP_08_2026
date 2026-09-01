import numpy as np, gc, sys, time, csv
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats2 import build, targets
import lightgbm as lgb
from datetime import date, timedelta
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
D0=date(2025,1,1); NU=250000; FINAL=408
M1,M2=2.3232887,10.7633307
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    return rmsle(z,A@c),float(c[0]),float(c[1])
base=dict(learning_rate=0.03,num_leaves=255,min_data_in_leaf=100,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
VAL=378
TR=[t for t in range(186,VAL-29,12)]
print("PHASE 1: calibrated early stopping",flush=True)
Xtr,names=build(TR); ytr=targets(TR); ztr=np.log1p(ytr); btr=(ytr>0).astype(np.int8)
Xva,_=build([VAL]); yva=targets([VAL]); zva=np.log1p(yva)
CK=list(range(50,1601,50))
md=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr,ztr,feature_name=names),num_boost_round=1600)
mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(Xtr,btr,feature_name=names),num_boost_round=1600)
pos=btr==1
mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr[pos],ztr[pos],feature_name=names),num_boost_round=1600)
del Xtr; gc.collect()
print(f"{'iter':>6} {'direct_raw':>11} {'direct_cal':>11} {'hurdle_cal':>11}",flush=True)
bd=(9,0); bh=(9,0)
Pd={};Pc={};Pr={}
for k in CK:
    pdk=md.predict(Xva,num_iteration=k); pck=mc.predict(Xva,num_iteration=k); prk=mr.predict(Xva,num_iteration=k)
    rd,_,_=cal(pdk,zva); rh,_,_=cal(pck*prk,zva)
    Pd[k]=pdk;Pc[k]=pck;Pr[k]=prk
    if rd<bd[0]: bd=(rd,k)
    if rh<bh[0]: bh=(rh,k)
    if k%200==0 or k<=200: print(f"{k:6d} {rmsle(zva,pdk):11.5f} {rd:11.5f} {rh:11.5f}",flush=True)
print(f"\nBEST direct: iter={bd[1]} cal={bd[0]:.5f}   BEST hurdle: iter={bh[1]} cal={bh[0]:.5f}",flush=True)
bb=(9,0,0)
for w in np.arange(0,1.01,0.1):
    r,a,b=cal(w*Pd[bd[1]]+(1-w)*(Pc[bh[1]]*Pr[bh[1]]),zva)
    if r<bb[0]: bb=(r,w,a)
print(f"BEST blend w={bb[1]:.1f} cal={bb[0]:.5f} slope={bb[2]:.4f}",flush=True)
np.save(W+'work/v3_cfg.npy',np.array([bd[1],bh[1],bb[1],bb[2]]))
del Xva; gc.collect()

print("\nPHASE 2: retrain incl. anchor 2026-01-14, predict 2026-02-13",flush=True)
TR2=TR+[VAL]
Xtr,_=build(TR2); ytr=targets(TR2); ztr=np.log1p(ytr); btr=(ytr>0).astype(np.int8)
sc=len(TR2)/len(TR)
nd,nh=int(bd[1]*sc),int(bh[1]*sc)
md=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr,ztr,feature_name=names),num_boost_round=nd)
mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(Xtr,btr,feature_name=names),num_boost_round=nh)
pos=btr==1
mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr[pos],ztr[pos],feature_name=names),num_boost_round=nh)
del Xtr; gc.collect()
Xp,_=build([FINAL])
w=bb[1]
zh=w*md.predict(Xp)+(1-w)*(mc.predict(Xp)*mr.predict(Xp))
del Xp; gc.collect()
np.save(W+'work/v3_zh_final.npy',zh)
a=bb[2]; b=M1-a*zh.mean()
zf=np.clip(a*zh+b,0,None)
print(f"final zh mean={zh.mean():.4f} -> a={a:.4f} b={b:+.4f} -> mean={zf.mean():.6f} (M1={M1:.6f})",flush=True)
uids=np.load(W+'work/mat/uids.npy')
def wr(name,z):
    p=np.expm1(np.clip(z,0,None))
    with open(W+'submissions/'+name,'w',newline='') as f:
        c=csv.writer(f); c.writerow(['user_id','predict'])
        for u,v in zip(uids,p): c.writerow([int(u),float(v)])
    print("wrote",name,"mean_log1p",np.log1p(p).mean(),flush=True)
wr('06_v3_levelM1.csv',zf)
print("DONE",flush=True)
