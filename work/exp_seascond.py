"""Fable block C: falsify season-conditioning WITHOUT spending submissions.
Hold out anchors whose target window covers the NY peak; train with and without
market-level covariates; see whether conditioning improves transfer to the held-out peak.
Block 3 covariates (market level inside the FEATURE windows) are fully known for
every anchor including the final one -- no circularity."""
import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats3, lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'; MD=W+'work/mat/'
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None); return rmsle(z,A@c)
# aggregate market curve, observed on all 409 days
gmv=np.load(MD+'gmv.npy'); m=gmv.sum(0).astype(np.float64); m/=m.mean(); del gmv
act=np.load(MD+'active.npy'); ma=act.sum(0).astype(np.float64); ma/=ma.mean(); del act
def wmean(c,a,b): a=max(a,0); return float(c[a:b+1].mean())
def cov(t):
    """market level inside each feature window (known for ANY anchor, incl. the final)"""
    return [wmean(m,t-29,t),wmean(m,t-89,t),wmean(m,t-179,t),
            wmean(ma,t-29,t),wmean(ma,t-89,t),wmean(ma,t-179,t),
            wmean(m,t-29,t)/max(wmean(m,t-179,t),1e-9)]
COVN=['mk_m30','mk_m90','mk_m180','mk_a30','mk_a90','mk_a180','mk_ratio']
# held-out: anchors whose 30d target window overlaps the NY peak (day 349..365)
def hits_ny(t): return not (t+30<349 or t+1>365)
ALL=[t for t in range(120,349,6)]
TRN=[t for t in ALL if not hits_ny(t)]; HLD=[t for t in ALL if hits_ny(t)]
print(f"train {len(TRN)} anchors, held-out(NY peak) {len(HLD)}: {HLD}",flush=True)
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=4,verbose=-1,max_bin=255)
NU=250000; FRAC=0.25                                  # subsample users per anchor (Fable block B)
rng=np.random.default_rng(0)
def build(anchors,withcov):
    Xs=[];ys=[]
    for t in anchors:
        X,names=feats3.build([t]); y=feats3.targets([t])
        idx=rng.choice(NU,int(NU*FRAC),replace=False)
        X=X[idx]; y=y[idx]
        if withcov:
            C=np.tile(np.array(cov(t),dtype=np.float32),(len(idx),1))
            X=np.hstack([X,C])
        Xs.append(X); ys.append(y); del X; gc.collect()
    nm=names+(COVN if withcov else [])
    return np.vstack(Xs),np.concatenate(ys),nm
CK=list(range(50,451,50))
res={}
for withcov in (False,True):
    t0=time.time()
    X,y,names=build(TRN,withcov); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
    md=lgb.train({**base,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=450)
    mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=450)
    mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=450)
    del X; gc.collect()
    scores=[]
    for t in HLD:
        Xv,_=feats3.build([t]); zv=np.log1p(feats3.targets([t]))
        if withcov: Xv=np.hstack([Xv,np.tile(np.array(cov(t),dtype=np.float32),(NU,1))])
        PD={k:md.predict(Xv,num_iteration=k) for k in CK}          # predict ONCE per k
        PH={k:mc.predict(Xv,num_iteration=k)*mr.predict(Xv,num_iteration=k) for k in CK}
        best=min(cal(w*PD[kd]+(1-w)*PH[kh],zv) for kd in CK for kh in CK for w in (0.0,0.3,0.5))
        del PD,PH
        scores.append(best); del Xv; gc.collect()
    res[withcov]=scores
    print(f"withcov={withcov}: held-out NY anchors -> {[round(s,5) for s in scores]}  mean {np.mean(scores):.5f}  ({time.time()-t0:.0f}s)",flush=True)
d=np.mean(res[False])-np.mean(res[True])
print(f"\nSEASON-CONDITIONING EFFECT on transfer to an unseen peak: {d:+.5f}")
print("positive => conditioning helps => full run justified; ~0 => direction dead, 0 submissions spent")
