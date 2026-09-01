"""Does a rising-season validation anchor rank configs the way the real target does?
Ground truth on the real target: 262f/stride12 = 1.651367  BEATS  159f/stride6 = 1.653399"""
import numpy as np, gc, sys, time
sys.path.insert(0,'/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work')
import feats3, lightgbm as lgb
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None); return rmsle(z,A@c)
KEEP=('gmv_','to_ord_','active_','to_cart_s30','to_cart_s180','searches_s30','searches_s180',
      'tenure','overdue','conv180','aov30','aov180','aov365','r_g_','r_o_','r_a_',
      'sh_cat180','sh_wknd90','act_months','buy_months','buy_weeks8','ord_per_act90','gmv_per_act90')
base=dict(learning_rate=0.05,num_leaves=127,min_data_in_leaf=200,feature_fraction=0.6,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=8,verbose=-1,max_bin=255)
CK=list(range(50,601,50))
_,allnames=feats3.build([378]); sel=[i for i,n in enumerate(allnames) if n.startswith(KEEP)]
# VAL anchors: 336 -> target 2025-12-03..2026-01-01 (rising, seas +4.55%)
#              378 -> target 2026-01-15..2026-02-13 (trough, what we used)
for VAL in (336,378):
    zv=np.log1p(feats3.targets([VAL]))
    Xf,_=feats3.build([VAL]); Xv_full=Xf; Xv_sel=np.ascontiguousarray(Xf[:,sel])
    for tag,stride,use_sel in (("262f/stride12",12,False),("159f/stride6",6,True)):
        TR=[t for t in range(186,VAL-29,stride)]
        if len(TR)<5: print(f"  {VAL} {tag}: too few anchors"); continue
        t0=time.time(); Xf2,_=feats3.build(TR)
        X=np.ascontiguousarray(Xf2[:,sel]) if use_sel else Xf2
        if use_sel: del Xf2
        nm=[allnames[i] for i in sel] if use_sel else allnames
        Xv=Xv_sel if use_sel else Xv_full
        y=feats3.targets(TR); z=np.log1p(y); b=(y>0).astype(np.int8); pos=b==1
        md=lgb.train({**base,'objective':'regression'},lgb.Dataset(X,z,feature_name=nm),num_boost_round=600)
        mc=lgb.train({**base,'objective':'binary'},lgb.Dataset(X,b,feature_name=nm),num_boost_round=600)
        mr=lgb.train({**base,'objective':'regression'},lgb.Dataset(X[pos],z[pos],feature_name=nm),num_boost_round=600)
        del X; gc.collect()
        PD={k:md.predict(Xv,num_iteration=k) for k in CK}; PH={k:mc.predict(Xv,num_iteration=k)*mr.predict(Xv,num_iteration=k) for k in CK}
        best=min(cal(w*PD[kd]+(1-w)*PH[kh],zv) for kd in CK for kh in CK for w in (0.0,0.2,0.3,0.4,0.5))
        print(f"VAL={VAL} anchors={len(TR):2d} {tag:14s} -> {best:.5f}   ({time.time()-t0:.0f}s)",flush=True)
        del PD,PH; gc.collect()
    del Xv_full,Xv_sel; gc.collect()
print("\nREAL TARGET says: 262f/stride12 (1.651367) BEATS 159f/stride6 (1.653399)")
