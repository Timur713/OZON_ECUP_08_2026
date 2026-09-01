import numpy as np
from datetime import date, timedelta
D='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/'
D0=date(2025,1,1)
gmv=np.load(D+'gmv.npy'); NU,ND=gmv.shape
cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:])
def w(a,b): return cs[:,b+1]-cs[:,a]
def idx(s): return (date.fromisoformat(s)-D0).days

print("=== C. Seasonal ratio of E[log1p(y)] : target-window vs naive-window, 2025 ===")
for lbl,(s,e) in {
    "naive  Jan15-Feb13'25":("2025-01-15","2025-02-13"),
    "TARGET Feb14-Mar15'25":("2025-02-14","2025-03-15"),
    "naive  Jan15-Feb13'26":("2026-01-15","2026-02-13"),
}.items():
    y=w(idx(s),idx(e)); Z=np.log1p(y)
    print(f"{lbl}: E[z]={Z.mean():.4f}  P(y>0)={(y>0).mean():.4f}  E[z|y>0]={Z[y>0].mean():.4f}  sqrt(E[z^2])={np.sqrt((Z**2).mean()):.4f}")
zn=np.log1p(w(idx("2025-01-15"),idx("2025-02-13"))).mean()
zt=np.log1p(w(idx("2025-02-14"),idx("2025-03-15"))).mean()
print(f"\n>>> 2025 seasonal LOG-SPACE ratio E[z]_target / E[z]_naive = {zt/zn:.4f}   (GMV-space ratio was 1.1628)")
pn=(w(idx("2025-01-15"),idx("2025-02-13"))>0).mean(); pt=(w(idx("2025-02-14"),idx("2025-03-15"))>0).mean()
print(f">>> P(y>0) ratio = {pt/pn:.4f}   ({pn:.4f} -> {pt:.4f})")

print("\n=== D. user_id  vs  first-activity date (cohort leak?) ===")
act=np.load(D+'active.npy')
first=np.argmax(act,axis=1).astype(np.float64)
ever=act.sum(axis=1)>0
uids=np.load(D+'uids.npy').astype(np.float64)
print("uid range:",uids.min(),uids.max(),"n",len(uids))
r=np.corrcoef(uids[ever],first[ever])[0,1]
print(f"corr(user_id, first_active_day) = {r:.4f}")
# decile table
q=np.quantile(uids,np.linspace(0,1,11))
print(f"{'uid decile':>12} {'mean_first_day':>15} {'%first>day300':>14} {'mean_gmv_tot':>13}")
tot=cs[:,-1]
for i in range(10):
    m=(uids>=q[i])&(uids<=q[i+1])&ever
    print(f"{i:12d} {first[m].mean():15.1f} {100*(first[m]>300).mean():14.2f} {tot[m].mean():13.1f}")
