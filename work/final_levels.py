import numpy as np
from datetime import date, timedelta
MD='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/'
D0=date(2025,1,1)
gmv=np.load(MD+'gmv.npy'); NU,ND=gmv.shape
cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:])
def w(a,b): return cs[:,b+1]-cs[:,a]
def dd(x): return (date.fromisoformat(x)-D0).days
def st(s,e):
    y=w(dd(s),dd(e)); z=np.log1p(y)
    return dict(P=(y>0).mean(),mu=z[y>0].mean(),Ez=z.mean())
A=st("2025-01-15","2025-02-13"); B=st("2025-02-14","2025-03-15"); C=st("2026-01-15","2026-02-13")
print(f"{'window':26s} {'P(y>0)':>8} {'E[z|y>0]':>9} {'E[z]':>8}")
for n,d in [("Jan15-Feb13 2025",A),("Feb14-Mar15 2025",B),("Jan15-Feb13 2026",C)]:
    print(f"{n:26s} {d['P']:8.4f} {d['mu']:9.4f} {d['Ez']:8.4f}")

print("\n--- extrapolation of P to Feb14-Mar15 2026 ---")
r_ratio=B['P']/A['P']; P1=C['P']*r_ratio
oA=A['P']/(1-A['P']); oB=B['P']/(1-B['P']); oC=C['P']/(1-C['P'])
oT=oC*(oB/oA); P2=oT/(1+oT)
print(f"  ratio method : P = {C['P']:.4f} x {r_ratio:.4f} = {P1:.4f}")
print(f"  odds  method : odds {oC:.4f} x {oB/oA:.4f} = {oT:.4f} -> P = {P2:.4f}")
mu_t=C['mu']*(B['mu']/A['mu'])
print(f"  E[z|y>0]     : {C['mu']:.4f} x {B['mu']/A['mu']:.4f} = {mu_t:.4f}")
print(f"\n  => E[z]_target  ratio-method = {P1*mu_t:.4f}")
print(f"  => E[z]_target  odds -method = {P2*mu_t:.4f}")
print(f"  => E[z]_target  direct YoY   = {B['Ez']*(C['Ez']/A['Ez']):.4f}")
lo,hi=min(P1*mu_t,P2*mu_t),max(P1*mu_t,B['Ez']*(C['Ez']/A['Ez']))
print(f"\n  RECOMMENDED L range = [{lo:.3f}, {hi:.3f}], point estimate {(lo+hi)/2:.3f}")
print(f"  RECOMMENDED P range = [{min(P1,P2):.4f}, {max(P1,P2):.4f}], point {(P1+P2)/2:.4f}")
print(f"\n  cost of being wrong by the full range width ({(hi-lo)/((lo+hi)/2)*100:.1f}%): ~{((hi-lo)/2/((lo+hi)/2))**2*10.3:.4f} MSE -> ~{np.sqrt(1.6454**2+((hi-lo)/2/((lo+hi)/2))**2*10.3)-1.6454:+.4f} RMSLE")
