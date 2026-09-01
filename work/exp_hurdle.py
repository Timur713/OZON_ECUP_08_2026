import numpy as np, gc, sys
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats import build, targets
import lightgbm as lgb
from datetime import date, timedelta
D0=date(2025,1,1); NU=250000
def rmsle(z,zh): return float(np.sqrt(np.mean((z-np.clip(zh,0,None))**2)))
V=378; TR=[t for t in range(210,V-29,17)]
Xtr,names=build(TR,verbose=False); ytr=targets(TR)
Xva,_=build([V],verbose=False); yva=targets([V])
ztr=np.log1p(ytr); zva=np.log1p(yva); btr=(ytr>0).astype(np.int8)
print("train",Xtr.shape,"pos rate",btr.mean())

base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.7,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,num_threads=8,verbose=-1,max_bin=127)
N=600
# 1) direct L2 on z
m1=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr,ztr,feature_name=names),num_boost_round=N)
p1=m1.predict(Xva)
# 2) hurdle: P(y>0) x E[z|y>0]
mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(Xtr,btr,feature_name=names),num_boost_round=N)
pos=btr==1
mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr[pos],ztr[pos],feature_name=names),num_boost_round=N)
pc=mc.predict(Xva); pr=mr.predict(Xva); p2=pc*pr
def cal(p,z):
    A=np.vstack([p,np.ones_like(p)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None); return rmsle(z,A@c),c
r1,c1=cal(p1,zva); r2,c2=cal(p2,zva)
print(f"\ndirect L2 : raw={rmsle(zva,p1):.4f}  calibrated={r1:.4f}  (a={c1[0]:.3f} b={c1[1]:.3f}) meanp={p1.mean():.4f}")
print(f"hurdle    : raw={rmsle(zva,p2):.4f}  calibrated={r2:.4f}  (a={c2[0]:.3f} b={c2[1]:.3f}) meanp={p2.mean():.4f}")
for w in [0.3,0.4,0.5,0.6,0.7]:
    pb=w*p1+(1-w)*p2; rb,_=cal(pb,zva)
    print(f"blend w_direct={w:.1f} : calibrated={rb:.4f}")
print(f"\nAUC-ish: pos-rate true={(yva>0).mean():.4f} pred-mean p={pc.mean():.4f}")
print(f"true E[z|y>0]={zva[yva>0].mean():.4f}  pred mean of pr={pr.mean():.4f}")
np.save('/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/p1.npy',p1)
np.save('/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/p2.npy',p2)
np.save('/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/zva378.npy',zva)
