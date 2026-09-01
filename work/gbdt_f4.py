"""GBDT on feats4 (304 features incl. the 6 never-used columns). Final model for the stack."""
import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats4, lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
VAL=378; FINAL=408
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
KD,KH,WB=250,250,0.3
TR=[t for t in range(186,VAL+1,12)]
sc=len(TR)/14.0; kd,kh=int(KD*sc),int(KH*sc)
t0=time.time()
X,names=feats4.build(TR); y=feats4.targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
print(f"anchors={len(TR)} X={X.shape} {X.nbytes/1e9:.2f}GB kd={kd} kh={kh} {time.time()-t0:.0f}s",flush=True)
Xp,_=feats4.build([FINAL])
PD=[];PH=[]
for seed in (1,2):
    P={**base,'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed}
    md=lgb.train({**P,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=kd)
    mc=lgb.train({**P,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=kh)
    mr=lgb.train({**P,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=kh)
    PD.append(md.predict(Xp)); PH.append(mc.predict(Xp)*mr.predict(Xp))
    if seed==1:
        imp=sorted(zip(names,mc.feature_importance('gain')),key=lambda x:-x[1])
        newf=[(n,g) for n,g in imp if n.startswith(('has_','search_to_cart','cat_to_cart'))][:6]
        print("  top NEW features in the P(buy) head:",[(n,int(g)) for n,g in newf],flush=True)
        print("  their rank among all:",[imp.index((n,g))+1 for n,g in newf],flush=True)
    print(f"  seed{seed} {time.time()-t0:.0f}s",flush=True)
del X; gc.collect()
zh=WB*np.mean(PD,0)+(1-WB)*np.mean(PH,0)
np.save(W+'work/gbdt_f4_final.npy',zh)
print(f"saved gbdt_f4_final.npy mean={zh.mean():.4f} sd={zh.std():.4f}\nDONE",flush=True)
