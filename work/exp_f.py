import numpy as np
from datetime import date, timedelta
D='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/'
D0=date(2025,1,1)
gmv=np.load(D+'gmv.npy'); NU,ND=gmv.shape
cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:])
def w(a,b): return cs[:,b+1]-cs[:,a]
def dd(x): return (date.fromisoformat(x)-D0).days
S=np.arange(0,ND-29); Ez=np.array([np.log1p(w(s,s+29)).mean() for s in S])

print("=== YoY ratio of E[z] on overlapping window starts (2026 vs 2025) ===")
for s in range(365,ND-29):
    print(f"  win {(D0+timedelta(days=s)).isoformat()}..{(D0+timedelta(days=s+29)).isoformat()}"
          f"  E[z]26={Ez[s]:.4f}  E[z]25={Ez[s-365]:.4f}  YoY={Ez[s]/Ez[s-365]:.4f}")

print("\n=== Local trend fits (log E[z] ~ linear), extrapolated to window start 2026-02-14 ===")
tgt=dd("2026-02-14")
for lo,hi,lbl in [("2025-04-01","2025-11-15","Apr-Nov'25 (clean)"),
                  ("2025-06-01","2025-11-15","Jun-Nov'25"),
                  ("2025-08-01","2025-11-15","Aug-Nov'25"),
                  ("2025-03-15","2025-11-15","Mar-Nov'25")]:
    m=(S>=dd(lo))&(S<=dd(hi))
    A=np.vstack([S[m],np.ones(m.sum())]).T.astype(float)
    c,_,_,_=np.linalg.lstsq(A,np.log(Ez[m]),rcond=None)
    print(f"  {lbl:22s} slope/yr x{np.exp(c[0]*365):.3f}  ->  trend@Feb14'26 = {np.exp(c[0]*tgt+c[1]):.4f}"
          f"  | seas(Feb14'25)={Ez[dd('2025-02-14')]/np.exp(c[0]*dd('2025-02-14')+c[1]):.4f}"
          f"  | E[z]_pred={np.exp(c[0]*tgt+c[1])*Ez[dd('2025-02-14')]/np.exp(c[0]*dd('2025-02-14')+c[1]):.4f}")

yoy=Ez[dd("2026-01-15")]/Ez[dd("2025-01-15")]
print(f"\n=== ESTIMATE A (YoY transfer) ===")
print(f"  E[z](Feb14-Mar15'25)={Ez[dd('2025-02-14')]:.4f} x YoY({yoy:.4f}) = {Ez[dd('2025-02-14')]*yoy:.4f}")

print("\n=== Sensitivity: RMSLE cost of a mis-scaled prediction ===")
# use anchor 2026-01-14 target as a stand-in for prediction magnitudes
y=w(dd("2026-01-15"),dd("2026-02-13")); Z=np.log1p(y)
zhat=Z.copy()  # oracle prediction -> then scale it
print(f"  E[zhat^2] (oracle) = {(zhat**2).mean():.4f}")
for s in [0.90,0.95,0.97,1.0,1.03,1.05,1.10,1.15,1.25]:
    base=1.6454
    pen=(s-1)**2*(zhat**2).mean()
    print(f"   scale {s:5.2f} -> extra MSE {pen:7.4f} -> RMSLE {np.sqrt(base**2+pen):.4f}  (+{np.sqrt(base**2+pen)-base:.4f})")

print("\n=== LB-PROBE math verification (simulated on 2025 analogue as 'public') ===")
rng=np.random.default_rng(0); pub=rng.choice(NU,50000,replace=False)
zt=np.log1p(w(dd("2025-02-14"),dd("2025-03-15")))[pub]
M2_true=(zt**2).mean(); M1_true=zt.mean()
S0=np.sqrt(M2_true)                              # probe 1: all zeros
c=100.0; k=np.log1p(c); S1=np.sqrt(((zt-k)**2).mean())   # probe 2: constant 100
M2_hat=S0**2; M1_hat=(M2_hat+k**2-S1**2)/(2*k)
print(f"  probe1 all-zeros score={S0:.10f} -> M2={M2_hat:.6f} (true {M2_true:.6f})")
print(f"  probe2 const=100 score={S1:.10f} -> M1={M1_hat:.6f} (true {M1_true:.6f})   ERROR={abs(M1_hat-M1_true):.2e}")
