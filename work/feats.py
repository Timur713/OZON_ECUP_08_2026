import numpy as np, gc
from datetime import date, timedelta
MD='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/'
D0=date(2025,1,1)
def dd(x): return (date.fromisoformat(x)-D0).days
MATS=['gmv','to_ord','to_cart','searches','active','gmv_cat','gmv_search','search','cat']
WIN=[7,14,30,60,90,180]

def build(anchors, verbose=True):
    """returns X (n_anchor*NU, F) float32, names, and (anchor_idx, user_row) index"""
    NU=250000
    cols={}; names=[]
    A=len(anchors)
    for m in MATS:
        M=np.load(MD+m+'.npy')
        cs=np.zeros((NU,M.shape[1]+1),dtype=np.float32); np.cumsum(M,axis=1,dtype=np.float32,out=cs[:,1:])
        for W in WIN:
            arr=np.empty((A,NU),dtype=np.float32)
            for ai,t in enumerate(anchors):
                a=max(0,t-W+1); arr[ai]=cs[:,t+1]-cs[:,a]
            cols[f'{m}_s{W}']=arr
        # nonzero-day counts
        nz=(M>0).astype(np.float32)
        csn=np.zeros((NU,M.shape[1]+1),dtype=np.float32); np.cumsum(nz,axis=1,dtype=np.float32,out=csn[:,1:])
        for W in [30,90,180]:
            arr=np.empty((A,NU),dtype=np.float32)
            for ai,t in enumerate(anchors):
                a=max(0,t-W+1); arr[ai]=csn[:,t+1]-csn[:,a]
            cols[f'{m}_d{W}']=arr
        # recency: days since last positive
        ar=np.arange(M.shape[1],dtype=np.int16)
        lastpos=np.maximum.accumulate(np.where(M>0,ar,np.int16(-1)),axis=1)
        arr=np.empty((A,NU),dtype=np.float32)
        for ai,t in enumerate(anchors): arr[ai]=t-lastpos[:,t]
        cols[f'{m}_rec']=arr
        del M,cs,nz,csn,lastpos; gc.collect()
        if verbose: print("  feat",m,flush=True)
    # tenure (first active day)
    act=np.load(MD+'active.npy'); ar=np.arange(act.shape[1],dtype=np.int16)
    firstpos=np.where(act>0,ar,np.int16(9999)).min(axis=1)
    arr=np.empty((A,NU),dtype=np.float32)
    for ai,t in enumerate(anchors): arr[ai]=np.clip(t-firstpos,-1,None)
    cols['tenure']=arr
    del act; gc.collect()
    # calendar
    for ai,t in enumerate(anchors): pass
    doy=np.empty((A,NU),dtype=np.float32)
    for ai,t in enumerate(anchors): doy[ai]=(D0+timedelta(days=int(t))).timetuple().tm_yday
    cols['anchor_doy']=doy
    # derived ratios
    eps=1e-3
    def R(a,b,nm): cols[nm]=(cols[a]/(cols[b]+eps)).astype(np.float32)
    R('gmv_s30','gmv_s90','r_gmv_30_90'); R('gmv_s7','gmv_s30','r_gmv_7_30')
    R('gmv_s30','gmv_s180','r_gmv_30_180')
    R('to_ord_s30','to_cart_s30','conv_30'); R('gmv_s30','to_ord_s30','aov_30')
    R('gmv_s180','to_ord_s180','aov_180'); R('to_ord_s90','active_d90','ord_per_actday_90')
    R('gmv_s90','active_d90','gmv_per_actday_90'); R('searches_s30','active_d30','srch_per_actday_30')
    R('gmv_cat_s180','gmv_s180','share_cat_180')
    names=list(cols.keys())
    X=np.empty((A*NU,len(names)),dtype=np.float32)
    for j,n in enumerate(names):
        X[:,j]=cols[n].reshape(-1)
    del cols; gc.collect()
    # log1p transform the heavy-tailed sums
    for j,n in enumerate(names):
        if '_s' in n and not n.startswith('r_'):
            np.log1p(np.clip(X[:,j],0,None),out=X[:,j])
    return X,names

def targets(anchors,H=30):
    NU=250000
    gmv=np.load(MD+'gmv.npy'); ND=gmv.shape[1]
    cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:])
    ys=[]
    for t in anchors:
        b=min(ND-1,t+H)
        ys.append(cs[:,b+1]-cs[:,t+1] if t+1<=b else np.zeros(NU))
    del gmv,cs; gc.collect()
    return np.concatenate(ys).astype(np.float64)
