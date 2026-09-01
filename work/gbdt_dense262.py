"""Isolate the anchor-density effect: FULL 262 features, stride 6.
We only ever tested 262f/s12 (real 1.651536) vs 159f/s6 (real 1.653399) -- the two
changes were confounded. This isolates density."""
import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats3, lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
VAL=378; FINAL=408
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
KD,KH,WB=250,250,0.3          # same iteration budget as the 262f/s12 model that is in the stack
TR=[t for t in range(186,VAL+1,10)]   # 20 anchors ~5.2GB, fits without swap   # dense, INCLUDING the val anchor (final model)
sc=len(TR)/14.0
kd,kh=int(KD*sc**0.5),int(KH*sc**0.5)
print(f"anchors={len(TR)} kd={kd} kh={kh}",flush=True)
t0=time.time()
X,names=feats3.build(TR); y=feats3.targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
print(f"X={X.shape} {X.nbytes/1e9:.2f}GB {time.time()-t0:.0f}s",flush=True)
Xp,_=feats3.build([FINAL])
PD=[];PH=[]
for seed in (1,2):
    P={**base,'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed}
    md=lgb.train({**P,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=kd)
    mc=lgb.train({**P,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=kh)
    mr=lgb.train({**P,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=kh)
    PD.append(md.predict(Xp)); PH.append(mc.predict(Xp)*mr.predict(Xp))
    print(f"  seed{seed} {time.time()-t0:.0f}s",flush=True)
del X; gc.collect()
zh=WB*np.mean(PD,0)+(1-WB)*np.mean(PH,0)
np.save(W+'work/gbdt_d262_final.npy',zh)
print(f"saved gbdt_d262_final.npy mean={zh.mean():.4f} sd={zh.std():.4f}\nDONE",flush=True)
