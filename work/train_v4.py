import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats2, feats3
import lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    return rmsle(z,A@c),float(c[0]),float(c[1])
VAL=378; TR=[t for t in range(186,VAL-29,12)]
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
N=600; CK=list(range(50,N+1,50))
out={}
for tag,mod in [('v3_210',feats2),('v4_262',feats3)]:
    t0=time.time()
    X,names=mod.build(TR); y=mod.targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8)
    Xv,_=mod.build([VAL]); yv=mod.targets([VAL]); zv=np.log1p(yv)
    print(f"\n=== {tag}: {X.shape} built {time.time()-t0:.0f}s ===",flush=True)
    md=lgb.train({**base,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=N)
    mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=N)
    pos=b==1
    mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=N)
    del X; gc.collect()
    PD={k:md.predict(Xv,num_iteration=k) for k in CK}
    PC={k:mc.predict(Xv,num_iteration=k) for k in CK}
    PR={k:mr.predict(Xv,num_iteration=k) for k in CK}
    best=None
    for kd in CK:
        for kh in CK:
            for w in (0.0,0.2,0.3,0.4,0.5):
                r,a,_=cal(w*PD[kd]+(1-w)*(PC[kh]*PR[kh]),zv)
                if best is None or r<best[0]: best=(r,kd,kh,w,a)
    print(f"{tag} BEST cal={best[0]:.5f}  kd={best[1]} kh={best[2]} w={best[3]} slope={best[4]:.4f}",flush=True)
    zh=best[3]*PD[best[1]]+(1-best[3])*(PC[best[2]]*PR[best[2]])
    out[tag]=zh
    np.save(W+f'work/{tag}_valpred.npy',zh)
    if tag=='v4_262':
        imp=sorted(zip(names,md.feature_importance('gain')),key=lambda x:-x[1])[:15]
        print("TOP v4:",[n for n,_ in imp],flush=True)
    del Xv; gc.collect()
zv=np.log1p(feats2.targets([VAL]))
cbz,_=np.load(W+'work/cb_val.npy')
print("\n=== BLENDS (calibrated) ===",flush=True)
for nm,v in [('v3_210',out['v3_210']),('v4_262',out['v4_262']),('catboost',cbz)]:
    print(f"  {nm:10s} {cal(v,zv)[0]:.5f}",flush=True)
import itertools
grid=np.arange(0,1.001,0.05)
bb=None
for w1 in grid:
    for w2 in grid:
        if w1+w2>1.0001: continue
        w3=1-w1-w2
        r,_,_=cal(w1*out['v3_210']+w2*out['v4_262']+w3*cbz,zv)
        if bb is None or r<bb[0]: bb=(r,w1,w2,w3)
print(f"  BEST 3-way: v3={bb[1]:.2f} v4={bb[2]:.2f} cb={bb[3]:.2f} -> {bb[0]:.5f}",flush=True)
np.save(W+'work/v4_blend.npy',np.array(bb))
print("DONE",flush=True)
