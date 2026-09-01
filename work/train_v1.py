import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats import build, targets, dd
import lightgbm as lgb
from datetime import date, timedelta
D0=date(2025,1,1)
NU=250000
TR=[210,227,244,261,278,295,312,329,346]
VA=[378]
print("train anchors",[ (D0+timedelta(days=t)).isoformat() for t in TR])
print("val anchor   ",[ (D0+timedelta(days=t)).isoformat() for t in VA])

t0=time.time()
Xtr,names=build(TR); ytr=targets(TR); print("Xtr",Xtr.shape,time.time()-t0)
Xva,_   =build(VA); yva=targets(VA); print("Xva",Xva.shape,time.time()-t0)
np.save('work/names.npy',np.array(names))

ztr=np.log1p(ytr); zva=np.log1p(yva)
def rmsle(z,zh): return float(np.sqrt(np.mean((z-np.clip(zh,0,None))**2)))

# per-anchor population level (for normalized target)
lev_tr=np.concatenate([np.full(NU,ztr[i*NU:(i+1)*NU].mean()) for i in range(len(TR))])
lev_va=zva.mean()
print("train levels",[round(float(ztr[i*NU:(i+1)*NU].mean()),4) for i in range(len(TR))],"| val level",round(float(lev_va),4))

P=dict(objective='regression',metric='l2',learning_rate=0.05,num_leaves=127,
       min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,
       lambda_l2=5.0,num_threads=8,verbose=-1,max_bin=127)

res={}
# --- A: raw z target
m=lgb.train(P,lgb.Dataset(Xtr,ztr,feature_name=names),num_boost_round=700,
            valid_sets=[lgb.Dataset(Xva,zva,feature_name=names)],
            callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(200)])
pa=m.predict(Xva,num_iteration=m.best_iteration); res['A_raw_z']=rmsle(zva,pa)
print("A raw-z          RMSLE",res['A_raw_z'],"best_iter",m.best_iteration)

# --- B: normalized target z/level, rescaled by TRUE val level (oracle level)
mb=lgb.train(P,lgb.Dataset(Xtr,ztr/lev_tr,feature_name=names),num_boost_round=m.best_iteration)
pb_n=mb.predict(Xva)
res['B_norm_oracle']=rmsle(zva,pb_n*lev_va)
print("B norm x ORACLE  RMSLE",res['B_norm_oracle'])
# B with last-train-level (no knowledge of val level)
res['B_norm_lasttrain']=rmsle(zva,pb_n*ztr[8*NU:9*NU].mean())
print("B norm x lasttr  RMSLE",res['B_norm_lasttrain'])

# --- calibration of A: optimal affine on val (oracle) vs none
Aq=np.vstack([pa,np.ones_like(pa)]).T
c,_,_,_=np.linalg.lstsq(Aq,zva,rcond=None)
res['A_affine_oracle']=rmsle(zva,Aq@c)
print(f"A + oracle affine (a={c[0]:.4f},b={c[1]:.4f}) RMSLE",res['A_affine_oracle'])
s=float((pa@zva)/(pa@pa))
res['A_scale_oracle']=rmsle(zva,pa*s)
print(f"A + oracle scale  (s={s:.4f})            RMSLE",res['A_scale_oracle'])
print("mean pred A",pa.mean(),"vs true",zva.mean())

np.save('work/pa_val.npy',pa); np.save('work/zva.npy',zva)
imp=sorted(zip(names,m.feature_importance('gain')),key=lambda x:-x[1])[:25]
print("\nTOP FEATURES:"); [print(f"  {n:28s} {g:,.0f}") for n,g in imp]
print("\nSUMMARY",res)
