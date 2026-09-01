import numpy as np
from datetime import date, timedelta
D='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/'
D0=date(2025,1,1)
gmv=np.load(D+'gmv.npy')                      # (NU,409)
NU,ND=gmv.shape
cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:])
def wsum(a,b):  # inclusive day idx
    return cs[:,b+1]-cs[:,a]
act=np.load(D+'active.npy'); csa=np.zeros((NU,ND+1),dtype=np.int32); np.cumsum(act,axis=1,out=csa[:,1:])
def wact(a,b): return csa[:,b+1]-csa[:,a]

rows=[]
for t in range(29,379,7):
    prev=wsum(t-29,t); y=wsum(t+1,t+30)
    X=np.log1p(prev); Z=np.log1p(y)
    A=np.vstack([X,np.ones_like(X)]).T
    (a,b),_,_,_=np.linalg.lstsq(A,Z,rcond=None)
    resid=Z-(a*X+b)
    rmsle_aff=np.sqrt(np.mean(resid**2))
    rmsle_nai=np.sqrt(np.mean((Z-X)**2))
    pos=(y>0).mean(); pos_prev=(prev>0).mean()
    mu=Z[y>0].mean() if pos>0 else 0
    rows.append((t,(D0+timedelta(days=t)).isoformat(),(D0+timedelta(days=t+1)).isoformat(),
                 (D0+timedelta(days=t+30)).isoformat(),a,b,Z.mean(),pos,mu,pos_prev,
                 rmsle_nai,rmsle_aff))
print(f"{'anchor':>11} {'tgt_start':>11} {'a':>7} {'b':>7} {'E[z]':>7} {'P(y>0)':>7} {'mu|y>0':>7} {'Pprev>0':>8} {'RMSLEnai':>9} {'RMSLEaff':>9}")
for r in rows:
    print(f"{r[1]:>11} {r[2]:>11} {r[4]:7.4f} {r[5]:7.4f} {r[6]:7.4f} {r[7]:7.4f} {r[8]:7.4f} {r[9]:8.4f} {r[10]:9.4f} {r[11]:9.4f}")

# RMSLE decomposition at last anchor
t=378; y=wsum(t+1,t+30); Z=np.log1p(y); p=(y>0).mean(); mu=Z[y>0].mean(); sd=Z[y>0].std()
print(f"\n=== decomposition at anchor {(D0+timedelta(days=t))} (target {(D0+timedelta(days=t+1))}..{(D0+timedelta(days=t+30))}) ===")
print(f"P(y>0)={p:.4f}  E[z|y>0]={mu:.4f}  sd(z|y>0)={sd:.4f}  E[z]={Z.mean():.4f}  sqrt(E[z^2])={np.sqrt((Z**2).mean()):.4f}")
print(f"if we knew p and mu perfectly per-population: var_bernoulli={p*(1-p)*mu**2:.4f} var_value={p*sd**2:.4f} -> floor RMSLE={np.sqrt(p*sd**2+p*(1-p)*mu**2):.4f}")
