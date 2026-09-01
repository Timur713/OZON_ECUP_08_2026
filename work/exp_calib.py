import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
from feats import build, targets
import lightgbm as lgb
from datetime import date, timedelta
D0=date(2025,1,1); NU=250000
def ds(t): return (D0+timedelta(days=int(t))).isoformat()
def rmsle(z,zh): return float(np.sqrt(np.mean((z-np.clip(zh,0,None))**2)))
P=dict(objective='regression',metric='l2',learning_rate=0.05,num_leaves=127,
       min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,
       lambda_l2=5.0,num_threads=8,verbose=-1,max_bin=127)

VALS=[336,357,378]
store={}
for V in VALS:
    TR=[t for t in range(210,V-29,17)]
    Xtr,names=build(TR,verbose=False); ytr=targets(TR)
    Xva,_=build([V],verbose=False); yva=targets([V])
    ztr=np.log1p(ytr); zva=np.log1p(yva)
    m=lgb.train(P,lgb.Dataset(Xtr,ztr,feature_name=names),num_boost_round=300)
    p=m.predict(Xva)
    A=np.vstack([p,np.ones_like(p)]).T
    c,_,_,_=np.linalg.lstsq(A,zva,rcond=None)
    store[V]=dict(p=p,z=zva,a=float(c[0]),b=float(c[1]),
                  raw=rmsle(zva,p),orc=rmsle(zva,A@c),
                  meanp=float(p.mean()),meanz=float(zva.mean()),ntr=len(TR))
    print(f"val {ds(V)} ntr={len(TR)}: raw={store[V]['raw']:.4f} oracleAff={store[V]['orc']:.4f} "
          f"(a={c[0]:.4f} b={c[1]:.4f}) meanpred={p.mean():.4f} meantrue={zva.mean():.4f}",flush=True)
    del Xtr,Xva; gc.collect()

print("\n=== Can we RECOVER the oracle gain without seeing the target? ===")
print("strategy: slope a taken from PREVIOUS validation anchor, intercept b set so mean(pred)=L_est")
for i in range(1,len(VALS)):
    V=VALS[i]; Vp=VALS[i-1]
    s=store[V]; a_prev=store[Vp]['a']
    L_true=s['meanz']
    for lbl,L in [("ORACLE level",L_true),("+3% err",L_true*1.03),("-3% err",L_true*0.97),
                  ("+8% err",L_true*1.08)]:
        b=L-a_prev*s['meanp']
        print(f"  val {ds(V)} | a_prev={a_prev:.4f} | {lbl:14s} L={L:.4f} -> RMSLE={rmsle(s['z'],a_prev*s['p']+b):.4f}"
              f"   (raw {s['raw']:.4f}, oracle {s['orc']:.4f})")
np.save('/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/calib_store.npy',
        np.array([[V,store[V]['a'],store[V]['b'],store[V]['raw'],store[V]['orc']] for V in VALS]))
