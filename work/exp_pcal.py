import numpy as np, sys
from scipy.optimize import brentq
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/'
import pickle
# rerun heads quickly, saving pc/pr this time
import gc
sys.path.insert(0,W); from feats import build, targets
import lightgbm as lgb
def rmsle(z,zh): return float(np.sqrt(np.mean((z-np.clip(zh,0,None))**2)))
V=378; TR=[t for t in range(210,V-29,17)]
Xtr,names=build(TR,verbose=False); ytr=targets(TR)
Xva,_=build([V],verbose=False); yva=targets([V])
ztr=np.log1p(ytr); zva=np.log1p(yva); btr=(ytr>0).astype(np.int8)
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.7,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,num_threads=8,verbose=-1,max_bin=127)
mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(Xtr,btr,feature_name=names),num_boost_round=600)
pos=btr==1
mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(Xtr[pos],ztr[pos],feature_name=names),num_boost_round=600)
pc=mc.predict(Xva); pr=mr.predict(Xva)
np.save(W+'pc.npy',pc); np.save(W+'pr.npy',pr)

P_true=(yva>0).mean(); print(f"true P={P_true:.4f}  raw mean p={pc.mean():.4f}  ratio={pc.mean()/P_true:.4f}")
def shift_to(p,target):
    lg=np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
    f=lambda d: (1/(1+np.exp(-(lg+d)))).mean()-target
    d=brentq(f,-5,5); return 1/(1+np.exp(-(lg+d))),d

print(f"\n{'method':38s} {'RMSLE':>8}")
print(f"{'uncalibrated hurdle':38s} {rmsle(zva,pc*pr):8.4f}")
A=np.vstack([pc*pr,np.ones(len(pr))]).T; c,_,_,_=np.linalg.lstsq(A,zva,rcond=None)
print(f"{'+ oracle affine on zhat':38s} {rmsle(zva,A@c):8.4f}")
for lbl,Pt in [("ORACLE P",P_true),("P +3%",P_true*1.03),("P -3%",P_true*0.97),("P +6%",P_true*1.06)]:
    pcal,d=shift_to(pc,Pt)
    print(f"{'+ logit-shift p to '+lbl:38s} {rmsle(zva,pcal*pr):8.4f}   (delta={d:+.4f})")
# combine: logit-shift p AND affine on the value head
pcal,_=shift_to(pc,P_true)
zh=pcal*pr; A2=np.vstack([zh,np.ones(len(zh))]).T; c2,_,_,_=np.linalg.lstsq(A2,zva,rcond=None)
print(f"{'+ logit-shift p THEN oracle affine':38s} {rmsle(zva,A2@c2):8.4f}  (a={c2[0]:.3f} b={c2[1]:.3f})")
