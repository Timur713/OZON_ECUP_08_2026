import numpy as np, gc, sys, csv, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats3 import build, targets
import lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1,M2=2.3232887,10.7633307
VAL=378; FINAL=408
TR=[t for t in range(186,VAL-29,12)]+[VAL]
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
KD,KH,Wb=250,250,0.3
sc=len(TR)/(len(TR)-1); KD,KH=int(KD*sc),int(KH*sc)
t0=time.time()
X,names=build(TR); y=targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
print("built",X.shape,time.time()-t0,flush=True)
Xp,_=build([FINAL]); print("pred built",flush=True)
PD=[];PH=[]
for seed in [1,2,3]:
    p={**base,'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed}
    md=lgb.train({**p,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=KD)
    mc=lgb.train({**p,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=KH)
    mr=lgb.train({**p,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=KH)
    PD.append(md.predict(Xp)); PH.append(mc.predict(Xp)*mr.predict(Xp))
    print(f"  seed {seed} {time.time()-t0:.0f}s",flush=True)
del X; gc.collect()
zh=Wb*np.mean(PD,0)+(1-Wb)*np.mean(PH,0)
np.save(W+'work/v4_zh_final.npy',zh)
# level-match to the exactly known M1, slope from val corrected by the observed +8% transfer bias
a=0.9479*1.08
v=np.clip(a*zh+(M1-a*zh.mean()),0,None)
np.save(W+'work/v4_v.npy',v); np.save(W+'work/v4_slope.npy',np.array([a]))
p=np.expm1(v); uids=np.load(W+'work/mat/uids.npy')
with open(W+'submissions/13_v4_levelM1.csv','w',newline='') as f:
    c=csv.writer(f); c.writerow(['user_id','predict'])
    for u,x in zip(uids,p): c.writerow([int(u),float(x)])
print(f"wrote 13_v4_levelM1.csv a={a:.4f} mean_log1p={np.log1p(p).mean():.6f} E[v^2]={np.mean(v*v):.6f}",flush=True)
# LY basis: same calendar window last year, for the exact-fit step
g=np.log1p(np.load(W+'work/mat/gmv.npy')[:,44:74].sum(1))
np.save(W+'work/ly_basis.npy',g)
print(f"LY basis (Feb14-Mar15 2025): mean={g.mean():.4f} nonzero={100*(g>0).mean():.1f}%",flush=True)
print("DONE",flush=True)
