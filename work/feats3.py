"""feats2 + lagged 30d profile, weekly profile, weekend share, activity spread."""
import numpy as np, gc
from datetime import date, timedelta
MD='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/'
D0=date(2025,1,1)
MATS=['gmv','to_ord','to_cart','searches','active','gmv_cat','gmv_search','search','cat',
      'search_to_ord','cat_to_ord']
WIN=[3,7,14,30,60,90,180,270,365]; DWIN=[7,30,90,180,365]
PROF={'gmv':12,'to_ord':12,'active':6}      # lagged 30d blocks
WK={'gmv':8,'active':8}                      # lagged 7d blocks

def build(anchors, verbose=False):
    NU=250000; A=len(anchors); cols={}; ar=None
    dow=np.array([(D0+timedelta(days=int(i))).weekday() for i in range(409)])
    wknd=(dow>=5).astype(np.float32)
    for m in MATS:
        M=np.load(MD+m+'.npy'); ND=M.shape[1]
        if ar is None: ar=np.arange(ND,dtype=np.int16)
        cs=np.zeros((NU,ND+1),dtype=np.float32); np.cumsum(M,axis=1,dtype=np.float32,out=cs[:,1:])
        for Wd in WIN:
            a=np.empty((A,NU),dtype=np.float32)
            for ai,t in enumerate(anchors): a[ai]=cs[:,t+1]-cs[:,max(0,t-Wd+1)]
            cols[f'{m}_s{Wd}']=a
        a=np.full((A,NU),np.nan,dtype=np.float32)
        for ai,t in enumerate(anchors):
            lo,hi=t+1-365,t+30-365
            if lo>=0: a[ai]=cs[:,hi+1]-cs[:,lo]
        cols[f'{m}_LY']=a
        if m in PROF:
            for k in range(PROF[m]):
                a=np.empty((A,NU),dtype=np.float32)
                for ai,t in enumerate(anchors):
                    hi=t-30*k; lo=max(0,hi-29)
                    a[ai]=(cs[:,hi+1]-cs[:,lo]) if hi>=0 else 0.0
                cols[f'{m}_m{k}']=a
        if m in WK:
            for k in range(WK[m]):
                a=np.empty((A,NU),dtype=np.float32)
                for ai,t in enumerate(anchors):
                    hi=t-7*k; lo=max(0,hi-6)
                    a[ai]=(cs[:,hi+1]-cs[:,lo]) if hi>=0 else 0.0
                cols[f'{m}_w{k}']=a
        nz=(M>0).astype(np.float32)
        csn=np.zeros((NU,ND+1),dtype=np.float32); np.cumsum(nz,axis=1,dtype=np.float32,out=csn[:,1:])
        for Wd in DWIN:
            a=np.empty((A,NU),dtype=np.float32)
            for ai,t in enumerate(anchors): a[ai]=csn[:,t+1]-csn[:,max(0,t-Wd+1)]
            cols[f'{m}_d{Wd}']=a
        if m in ('gmv','active'):
            we=(M>0).astype(np.float32)*wknd[None,:]
            csw=np.zeros((NU,ND+1),dtype=np.float32); np.cumsum(we,axis=1,dtype=np.float32,out=csw[:,1:])
            a=np.empty((A,NU),dtype=np.float32)
            for ai,t in enumerate(anchors):
                lo=max(0,t-89); a[ai]=csw[:,t+1]-csw[:,lo]
            cols[f'{m}_wknd90']=a; del we,csw
        L1=np.maximum.accumulate(np.where(M>0,ar,np.int16(-1)),axis=1)
        Ls=np.concatenate([np.full((NU,1),np.int16(-1)),L1[:,:-1]],axis=1)
        rec=np.empty((A,NU),dtype=np.float32); gap=np.empty((A,NU),dtype=np.float32)
        idx=np.arange(NU)
        for ai,t in enumerate(anchors):
            l1=L1[:,t]; rec[ai]=t-l1
            l2=Ls[idx,np.clip(l1,0,None)]
            g=(l1-l2).astype(np.float32); g[l1<0]=np.nan; g[l2<0]=np.nan; gap[ai]=g
        cols[f'{m}_rec']=rec; cols[f'{m}_lastgap']=gap
        del M,cs,nz,csn,L1,Ls; gc.collect()
        if verbose: print("  ",m,flush=True)
    act=np.load(MD+'active.npy')
    fp=np.where(act>0,ar,np.int16(9999)).min(axis=1)
    a=np.empty((A,NU),dtype=np.float32)
    for ai,t in enumerate(anchors): a[ai]=np.clip(t-fp,-1,None)
    cols['tenure']=a; del act; gc.collect()
    a=np.empty((A,NU),dtype=np.float32)
    for ai,t in enumerate(anchors): a[ai]=(D0+timedelta(days=int(t))).timetuple().tm_yday
    cols['anchor_doy']=a
    # activity spread from the lagged profiles
    cols['act_months']=sum((cols[f'active_m{k}']>0).astype(np.float32) for k in range(PROF['active']))
    cols['buy_months']=sum((cols[f'gmv_m{k}']>0).astype(np.float32) for k in range(PROF['gmv']))
    cols['buy_weeks8']=sum((cols[f'gmv_w{k}']>0).astype(np.float32) for k in range(WK['gmv']))
    eps=1e-3
    def R(x,y,n): cols[n]=(cols[x]/(cols[y]+eps)).astype(np.float32)
    R('gmv_s30','gmv_s90','r_g_30_90'); R('gmv_s7','gmv_s30','r_g_7_30')
    R('gmv_s30','gmv_s180','r_g_30_180'); R('gmv_s90','gmv_s365','r_g_90_365')
    R('to_ord_s30','to_ord_s180','r_o_30_180'); R('active_d30','active_d180','r_a_30_180')
    R('to_ord_s30','to_cart_s30','conv30'); R('to_ord_s180','to_cart_s180','conv180')
    R('gmv_s30','to_ord_s30','aov30'); R('gmv_s180','to_ord_s180','aov180')
    R('gmv_s365','to_ord_s365','aov365'); R('gmv_s90','active_d90','gmv_per_act90')
    R('to_ord_s90','active_d90','ord_per_act90'); R('searches_s30','active_d30','srch_per_act30')
    R('gmv_cat_s180','gmv_s180','sh_cat180'); R('gmv_search_s180','gmv_s180','sh_srch180')
    R('gmv_LY','gmv_s30','r_LY_30'); R('gmv_wknd90','gmv_d90','sh_wknd90')
    for Wd in [90,180,365]:
        cols[f'overdue{Wd}']=(cols['gmv_rec']*cols[f'gmv_d{Wd}']/Wd).astype(np.float32)
    cols['overdue_gap']=(cols['gmv_rec']/(cols['gmv_lastgap']+1.0)).astype(np.float32)
    names=list(cols.keys())
    X=np.empty((A*NU,len(names)),dtype=np.float32)
    for j,n in enumerate(names): X[:,j]=cols[n].reshape(-1)
    del cols; gc.collect()
    for j,n in enumerate(names):
        if any(k in n for k in ('_s','_LY','_m','_w')) and not n.startswith(('r_','sh_','conv','aov','act_','buy_','overdue')):
            np.log1p(np.clip(X[:,j],0,None),out=X[:,j])
    return X,names

def targets(anchors,H=30):
    NU=250000; gmv=np.load(MD+'gmv.npy'); ND=gmv.shape[1]
    cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:])
    ys=[cs[:,min(ND-1,t+H)+1]-cs[:,t+1] for t in anchors]
    del gmv,cs; gc.collect(); return np.concatenate(ys)
