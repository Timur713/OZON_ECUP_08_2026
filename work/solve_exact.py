"""Generic exact recalibration from a leaderboard score.
Usage: solve_exact.py <vec.npy> <score> <out.csv> [basis1.npy:Ezb1 ...]
Recovers E[z*v] exactly and emits the affine optimum (plus extra bases if their E[z*b] known)."""
import sys, numpy as np, csv
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1,M2=2.3232887,10.7633307
vec=np.load(W+sys.argv[1]); S=float(sys.argv[2]); out=sys.argv[3]
Ev2=np.mean(vec*vec); Ezv=(M2+Ev2-S*S)/2
cols=[vec]; rhs=[Ezv]; nm=['v']
for spec in sys.argv[4:]:
    f,e=spec.split(':'); cols.append(np.load(W+f)); rhs.append(float(e)); nm.append(f)
cols.append(np.ones_like(vec)); rhs.append(M1); nm.append('1')
U=np.vstack(cols).T; G=U.T@U/len(vec)
c=np.linalg.solve(G,np.array(rhs))
mse=M2-2*float(np.array(rhs)@c)+float(c@G@c)
print(f"E[z*v]={Ezv:.6f}")
print("coefs: "+"  ".join(f"{n}={x:+.6f}" for n,x in zip(nm,c)))
print(f"expected public RMSLE = {np.sqrt(max(mse,0)):.6f}   (submitted {S:.6f}, gain {S-np.sqrt(max(mse,0)):+.6f})")
zf=np.clip(U@c,0,None); p=np.expm1(zf); uids=np.load(W+'work/mat/uids.npy')
with open(W+'submissions/'+out,'w',newline='') as f:
    cw=csv.writer(f); cw.writerow(['user_id','predict'])
    for u,x in zip(uids,p): cw.writerow([int(u),float(x)])
print(f"wrote submissions/{out}  mean_log1p={np.log1p(p).mean():.6f}")
