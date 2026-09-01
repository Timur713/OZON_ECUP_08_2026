"""IDEA 2: distributional estimate. E[z] = integral of P(z>t) dt.
Train K binary models for P(z > t_k) and integrate. Structurally different from
both the direct regressor and the p*mu hurdle -> genuine stack diversity."""
import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats4, lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
VAL=378; FINAL=408
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=7,verbose=-1,max_bin=255)
TR=[t for t in range(186,VAL+1,12)]
NIT=int(250*len(TR)/14.0)
t0=time.time()
X,names=feats4.build(TR); y=feats4.targets(TR); z=np.log1p(y)
zp=z[z>0]
TH=np.concatenate([[0.0],np.quantile(zp,np.linspace(0.08,0.94,11))])
print(f"thresholds: {np.round(TH,3)}",flush=True)
Xp,_=feats4.build([FINAL])
Sv=np.zeros((len(TH),250000))
for i,t in enumerate(TH):
    lab=(z>t).astype(np.int8)
    m=lgb.train({**base,'objective':'binary'},lgb.Dataset(X,lab,feature_name=names),num_boost_round=NIT)
    Sv[i]=m.predict(Xp)
    print(f"  P(z>{t:.3f}) base-rate={lab.mean():.4f}  pred-mean={Sv[i].mean():.4f}  {time.time()-t0:.0f}s",flush=True)
del X; gc.collect()
# E[z] = sum over slabs of P(z>t)*dt, trapezoid, with a tail beyond the last threshold
edges=np.append(TH,TH[-1]+ (TH[-1]-TH[-2]))
zh=np.zeros(250000)
for i in range(len(TH)):
    w=edges[i+1]-edges[i]
    zh+=Sv[i]*w
np.save(W+'work/gbdt_surv_final.npy',zh); np.save(W+'work/gbdt_surv_S.npy',Sv)
print(f"saved gbdt_surv_final.npy mean={zh.mean():.4f} sd={zh.std():.4f}\nDONE",flush=True)
