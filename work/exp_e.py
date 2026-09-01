import numpy as np
from datetime import date, timedelta
D='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/'
D0=date(2025,1,1)
gmv=np.load(D+'gmv.npy'); NU,ND=gmv.shape
cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:])
def w(a,b): return cs[:,b+1]-cs[:,a]

# E[z] and P(y>0) for every 30-day window START s (window = [s, s+29])
S=np.arange(0,ND-29)
Ez=np.zeros(len(S)); P=np.zeros(len(S)); Mu=np.zeros(len(S))
for i,s in enumerate(S):
    y=w(s,s+29); Z=np.log1p(y); Ez[i]=Z.mean(); P[i]=(y>0).mean(); Mu[i]=Z[y>0].mean()

def dt(s): return (D0+timedelta(days=int(s)))
# log-linear trend fit on log(Ez), excluding holiday-affected windows
lg=np.log(Ez)
# windows overlapping strong holiday periods (late Nov-Dec BF/NY, mid-Feb..mid-Mar gifting)
def overlaps(s,a,b):
    return not (s+29 < a or s > b)
def dd(x): return (date.fromisoformat(x)-D0).days
hol=[(dd("2025-11-15"),dd("2026-01-10")),(dd("2025-02-05"),dd("2025-03-12"))]
mask_clean=np.array([not any(overlaps(s,a,b) for a,b in hol) for s in S])
A=np.vstack([S,np.ones_like(S)]).T.astype(float)
coef,_,_,_=np.linalg.lstsq(A[mask_clean],lg[mask_clean],rcond=None)
trend=A@coef
seas=lg-trend
print(f"log-linear trend slope = {coef[0]*365:.4f} per year -> x{np.exp(coef[0]*365):.3f}/yr on E[z]")

print(f"\n{'win_start':>11} {'win_end':>11} {'E[z]':>7} {'P(y>0)':>7} {'mu':>6} {'trend':>7} {'seas%':>7}")
for i,s in enumerate(S):
    if s%14: continue
    print(f"{dt(s).isoformat():>11} {dt(s+29).isoformat():>11} {Ez[i]:7.4f} {P[i]:7.4f} {Mu[i]:6.3f} {np.exp(trend[i]):7.4f} {100*(np.exp(seas[i])-1):+7.2f}")

# seasonal factor for the TARGET window (Feb14-Mar15) taken from 2025
s25=dd("2025-02-14"); i25=list(S).index(s25)
sn25=dd("2025-01-15"); j25=list(S).index(sn25)
sn26=dd("2026-01-15"); j26=list(S).index(sn26)
print(f"\nseasonal factor Feb14-Mar15'25  = {np.exp(seas[i25]):.4f}")
print(f"seasonal factor Jan15-Feb13'25  = {np.exp(seas[j25]):.4f}")
print(f"seasonal factor Jan15-Feb13'26  = {np.exp(seas[j26]):.4f}")
print(f"PURE seasonal ratio (target/naive, 2025) = {np.exp(seas[i25]-seas[j25]):.4f}")

# extrapolate to target window Feb14-Mar15 2026
s26=dd("2026-02-14")
trend26=coef[0]*s26+coef[1]
pred_Ez = np.exp(trend26+seas[i25])
print(f"\n>>> extrapolated trend at 2026-02-14 = {np.exp(trend26):.4f}")
print(f">>> PREDICTED E[z] for target window Feb14-Mar15 2026 = {pred_Ez:.4f}")
print(f"    (naive Jan15-Feb13'26 actual E[z] = {Ez[j26]:.4f} -> implied uplift x{pred_Ez/Ez[j26]:.4f})")
np.save('/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/Ez_windows.npy',np.vstack([S,Ez,P,Mu,trend,seas]))
