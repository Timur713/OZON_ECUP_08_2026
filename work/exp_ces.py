import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats2 import build, targets
import lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    return rmsle(z,A@c),float(c[0]),float(c[1])
VAL=378; TR=[t for t in range(186,VAL-29,12)]
X,names=build(TR); y=targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8)
Xv,_=build([VAL]); yv=targets([VAL]); zv=np.log1p(yv)
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
N=900; CK=list(range(50,N+1,50))
print("training direct...",flush=True)
md=lgb.train({**base,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=N)
print("training binary...",flush=True)
mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=N)
pos=b==1
print("training cond-value...",flush=True)
mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=N)
del X; gc.collect()
print(f"\n{'iter':>5} {'direct_raw':>11} {'direct_cal':>11} {'hurdle_cal':>11} {'blend_cal':>10}",flush=True)
best=(9,)
D={};C={};R={}
for k in CK:
    D[k]=md.predict(Xv,num_iteration=k); C[k]=mc.predict(Xv,num_iteration=k); R[k]=mr.predict(Xv,num_iteration=k)
    rd,_,_=cal(D[k],zv); rh,_,_=cal(C[k]*R[k],zv); rb,ab,_=cal(0.5*D[k]+0.5*C[k]*R[k],zv)
    if rb<best[0]: best=(rb,k,ab)
    print(f"{k:5d} {rmsle(zv,D[k]):11.5f} {rd:11.5f} {rh:11.5f} {rb:10.5f}",flush=True)
print(f"\nBEST blend@50/50: iter={best[1]} cal={best[0]:.5f} slope={best[2]:.4f}",flush=True)
# free search over (kd, kh, w)
bb=(9,)
for kd in CK:
    for kh in CK:
        for w in [0.3,0.4,0.5,0.6,0.7]:
            r,a,_=cal(w*D[kd]+(1-w)*(C[kh]*R[kh]),zv)
            if r<bb[0]: bb=(r,kd,kh,w,a)
print(f"BEST free: kd={bb[1]} kh={bb[2]} w={bb[3]} cal={bb[0]:.5f} slope={bb[4]:.4f}",flush=True)
np.save(W+'work/ces_best.npy',np.array(bb))
