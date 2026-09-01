import numpy as np
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/'
z=np.load(W+'zva378.npy'); pc=np.load(W+'pc.npy'); pr=np.load(W+'pr.npy')
zh=pc*pr
A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None)
e2=(z-(A@c))**2
N=len(z); full=np.sqrt(e2.mean())
print(f"full-population RMSLE={full:.6f}  mean e2={e2.mean():.4f}  sd e2={e2.std():.4f}")
rng=np.random.default_rng(1)
for n,lbl in [(50000,'PUBLIC  (50k)'),(200000,'PRIVATE (200k)')]:
    s=np.array([np.sqrt(e2[rng.choice(N,n,replace=False)].mean()) for _ in range(400)])
    print(f"{lbl}: sd of reported RMSLE = {s.std():.5f}   95% range +-{1.96*s.std():.5f}")
# how big a TRUE improvement is needed to reliably win
print("\nLB spread #1->#50 on the real board = 0.0063")
print("=> public-LB differences below ~0.010 between teams are mostly sampling noise")
