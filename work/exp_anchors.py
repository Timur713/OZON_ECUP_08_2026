"""Controlled test: does anchor density help the GBDT? Same features, different stride."""
import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats3
import lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
VAL=378
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None); return rmsle(z,A@c)
KEEP=('gmv_','to_ord_','active_','to_cart_s30','to_cart_s180','searches_s30','searches_s180',
      'tenure','overdue','conv180','aov30','aov180','aov365','r_g_','r_o_','r_a_',
      'sh_cat180','sh_wknd90','act_months','buy_months','buy_weeks8','ord_per_act90','gmv_per_act90')
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=6,verbose=-1,max_bin=255)
CK=list(range(50,501,50))
res={}
_,allnames=feats3.build([VAL])
sel=[i for i,n in enumerate(allnames) if n.startswith(KEEP)]
print(f"kept {len(sel)} of {len(allnames)} features",flush=True)
Xv_full,_=feats3.build([VAL]); Xv=np.ascontiguousarray(Xv_full[:,sel]); del Xv_full; gc.collect()
zv=np.log1p(feats3.targets([VAL]))
names=[allnames[i] for i in sel]
for stride in (12,8):
    TR=[t for t in range(186,VAL-29,stride)]
    t0=time.time()
    Xf,_=feats3.build(TR); X=np.ascontiguousarray(Xf[:,sel]); del Xf; gc.collect()
    y=feats3.targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
    print(f"stride={stride} anchors={len(TR)} X={X.shape} {X.nbytes/1e9:.2f}GB built {time.time()-t0:.0f}s",flush=True)
    md=lgb.train({**base,'objective':'regression'},lgb.Dataset(X,z,feature_name=names),num_boost_round=500)
    mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(X,b,feature_name=names),num_boost_round=500)
    mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=names),num_boost_round=500)
    del X; gc.collect()
    PD={k:md.predict(Xv,num_iteration=k) for k in CK}
    PH={k:mc.predict(Xv,num_iteration=k)*mr.predict(Xv,num_iteration=k) for k in CK}
    best=None
    for kd in CK:
        for kh in CK:
            for w in (0.0,0.2,0.3,0.4,0.5):
                r=cal(w*PD[kd]+(1-w)*PH[kh],zv)
                if best is None or r<best[0]: best=(r,kd,kh,w)
    res[stride]=best
    np.save(W+f'work/anchors_s{stride}_val.npy',best[3]*PD[best[1]]+(1-best[3])*PH[best[2]])
    print(f"  stride={stride}: cal={best[0]:.5f} kd={best[1]} kh={best[2]} w={best[3]}  ({time.time()-t0:.0f}s)",flush=True)
print(f"\nANCHOR DENSITY EFFECT: stride12={res[12][0]:.5f}  stride8={res[8][0]:.5f}  gain={res[12][0]-res[8][0]:+.5f}",flush=True)
