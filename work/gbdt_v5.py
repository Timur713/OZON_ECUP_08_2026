"""Pruned features + dense anchors -> new GBDT base for the stack."""
import numpy as np, gc, sys, time, csv
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats3, lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1=2.3232887; VAL=378; FINAL=408
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None); return rmsle(z,A@c)
KEEP=('gmv_','to_ord_','active_','to_cart_s30','to_cart_s180','searches_s30','searches_s180',
      'tenure','overdue','conv180','aov30','aov180','aov365','r_g_','r_o_','r_a_',
      'sh_cat180','sh_wknd90','act_months','buy_months','buy_weeks8','ord_per_act90','gmv_per_act90')
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
CK=list(range(50,701,50))
_,allnames=feats3.build([VAL]); sel=[i for i,n in enumerate(allnames) if n.startswith(KEEP)]
names=[allnames[i] for i in sel]; print(f"{len(sel)} features",flush=True)
Xf,_=feats3.build([VAL]); Xv=np.ascontiguousarray(Xf[:,sel]); del Xf; gc.collect()
zv=np.log1p(feats3.targets([VAL]))
STRIDE=6
TR=[t for t in range(186,VAL-29,STRIDE)]
t0=time.time(); Xf,_=feats3.build(TR); X=np.ascontiguousarray(Xf[:,sel]); del Xf; gc.collect()
y=feats3.targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
print(f"stride={STRIDE} anchors={len(TR)} X={X.shape} {X.nbytes/1e9:.2f}GB {time.time()-t0:.0f}s",flush=True)
md=lgb.train({**base,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=700)
mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=700)
mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=700)
del X; gc.collect()
PD={k:md.predict(Xv,num_iteration=k) for k in CK}; PH={k:mc.predict(Xv,num_iteration=k)*mr.predict(Xv,num_iteration=k) for k in CK}
best=None
for kd in CK:
    for kh in CK:
        for w in (0.0,0.2,0.3,0.4,0.5):
            r=cal(w*PD[kd]+(1-w)*PH[kh],zv)
            if best is None or r<best[0]: best=(r,kd,kh,w)
print(f"stride6/159f: cal={best[0]:.5f} kd={best[1]} kh={best[2]} w={best[3]}   (s8 was 1.67070, s12 1.67128)",flush=True)
np.save(W+'work/gbdt_v5_val.npy',best[3]*PD[best[1]]+(1-best[3])*PH[best[2]])
del PD,PH,Xv; gc.collect()
# final: retrain incl. VAL anchor, predict at 408
TR2=TR+[VAL]; sc=len(TR2)/len(TR); kd,kh=int(best[1]*sc),int(best[2]*sc)
Xf,_=feats3.build(TR2); X=np.ascontiguousarray(Xf[:,sel]); del Xf; gc.collect()
y=feats3.targets(TR2); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
Xf,_=feats3.build([FINAL]); Xp=np.ascontiguousarray(Xf[:,sel]); del Xf; gc.collect()
PDs=[];PHs=[]
for seed in (1,2):
    P={**base,'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed}
    md=lgb.train({**P,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=kd)
    mc=lgb.train({**P,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=kh)
    mr=lgb.train({**P,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=kh)
    PDs.append(md.predict(Xp)); PHs.append(mc.predict(Xp)*mr.predict(Xp))
    print(f"  seed{seed} {time.time()-t0:.0f}s",flush=True)
zh=best[3]*np.mean(PDs,0)+(1-best[3])*np.mean(PHs,0)
np.save(W+'work/gbdt_v5_final.npy',zh)
print(f"saved gbdt_v5_final.npy mean={zh.mean():.4f} sd={zh.std():.4f}",flush=True)
print("DONE",flush=True)
