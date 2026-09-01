import numpy as np, gc, sys, csv, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats2 import build, targets
import lightgbm as lgb
from datetime import date, timedelta
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
D0=date(2025,1,1); NU=250000; FINAL=408; VAL=378
M1,M2=2.3232887,10.7633307
base=dict(learning_rate=0.03,num_leaves=255,min_data_in_leaf=100,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
TR=[t for t in range(186,VAL-29,12)]+[VAL]
print("anchors:",len(TR),flush=True)
t0=time.time()
X,names=build(TR); y=targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8)
print("built",X.shape,time.time()-t0,flush=True)
sc=len(TR)/(len(TR)-1)
ND,NC,NR=int(78*sc),int(137*sc),int(78*sc)
pos=b==1
PD=[];PC=[];PR=[]
Xp,_=build([FINAL]); print("pred matrix built",Xp.shape,flush=True)
for seed in [1,2]:
    p={**base,'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed}
    md=lgb.train({**p,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=ND)
    mc=lgb.train({**p,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=NC)
    mr=lgb.train({**p,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=NR)
    PD.append(md.predict(Xp)); PC.append(mc.predict(Xp)); PR.append(mr.predict(Xp))
    print(f"  seed {seed} done {time.time()-t0:.0f}s",flush=True)
pd_=np.mean(PD,0); pc=np.mean(PC,0); pr=np.mean(PR,0)
zh=0.5*pd_+0.5*(pc*pr)
np.save(W+'work/v2f_zh.npy',zh)
print(f"zh mean={zh.mean():.4f} sd={zh.std():.4f}",flush=True)
uids=np.load(W+'work/mat/uids.npy')
def wr(name,a):
    bb=M1-a*zh.mean(); zf=np.clip(a*zh+bb,0,None); p=np.expm1(zf)
    with open(W+'submissions/'+name,'w',newline='') as f:
        c=csv.writer(f); c.writerow(['user_id','predict'])
        for u,v in zip(uids,p): c.writerow([int(u),float(v)])
    print(f"wrote {name}  a={a:.4f} b={bb:+.4f} mean_log1p={np.log1p(p).mean():.6f} Ezh2={np.mean(zf**2):.6f}",flush=True)
    return np.mean(zf**2)
e1=wr('06_v2f_a1027.csv',1.027)
e2=wr('07_v2f_a1065.csv',1.065)
np.save(W+'work/v2f_Ezh2.npy',np.array([e1,e2]))
print("DONE",flush=True)
