"""Fable: residual reliability curve. Out-of-fold model on each of 13 historical
windows -> residuals -> corr(r_j, r_k) by gap.
 ~0 everywhere  => no repeatable structure left => practical ceiling reached
 flat positive  => missed stable structure
 decaying       => phase readable but undercaptured
Zero submissions."""
import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats3, lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
NU=250000; FRAC=0.30; ND=409
starts=list(range(ND-30,-1,-30))[::-1]          # 13 non-overlapping windows
anchors=[s-1 for s in starts]
USE=[i for i,t in enumerate(anchors) if t>=120] # need enough history for the features
print(f"windows {len(starts)}, usable anchors {len(USE)}: {[anchors[i] for i in USE]}",flush=True)
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=7,verbose=-1,max_bin=255)
NIT=300
rng=np.random.default_rng(0)
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
RES={}; t0=time.time()
for k in USE:
    tk=anchors[k]
    # OUT-OF-FOLD: train only on anchors whose 30d target window does not overlap window k
    TR=[t for t in range(120,349,12) if (t+30 < starts[k]) or (t+1 > starts[k]+29)]
    Xs=[];ys=[]
    for t in TR:
        X,names=feats3.build([t]); y=feats3.targets([t])
        idx=rng.choice(NU,int(NU*FRAC),replace=False)
        Xs.append(X[idx]); ys.append(y[idx]); del X; gc.collect()
    X=np.vstack(Xs); y=np.concatenate(ys); del Xs,ys; gc.collect()
    z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
    mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=NIT)
    mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=NIT)
    del X; gc.collect()
    Xv,_=feats3.build([tk]); zv=np.log1p(feats3.targets([tk]))
    zh=mc.predict(Xv)*mr.predict(Xv); del Xv; gc.collect()
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,zv,rcond=None)
    p=A@c; RES[k]=zv-p
    print(f"  window {k} anchor {tk}: ntrain={len(TR)} rmsle={rmsle(zv,p):.5f}  ({time.time()-t0:.0f}s)",flush=True)
    np.save(W+'work/residcurve.npy',np.array([[kk]+list(v) for kk,v in RES.items()],dtype=np.float32)[:, :1])
ks=sorted(RES)
print(f"\n{'gap':>6} {'pairs':>6} {'corr(resid)':>12}")
out=[]
for gi in range(1,len(ks)):
    cs=[float(np.corrcoef(RES[ks[i]],RES[ks[i+gi]])[0,1]) for i in range(len(ks)-gi)]
    print(f"{gi*30:6d} {len(cs):6d} {np.mean(cs):12.4f}")
    out.append((gi*30,float(np.mean(cs))))
np.save(W+'work/resid_corr.npy',np.array(out))
R=np.stack([RES[k] for k in ks])
np.save(W+'work/resid_mat.npy',R.astype(np.float32))
print("\nreference: corr of RAW z between windows was 0.58 (g=30) .. 0.44 (g=330)")
print("DONE",flush=True)
