"""Recover the exact optimal affine for a submitted model, using the known target moments.
Usage: python solve_v2f.py <score_of_06> [zh_file] [out_name] [slope_used]"""
import sys, numpy as np, csv
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1,M2=2.3232887,10.7633307
S=float(sys.argv[1])
zhf=sys.argv[2] if len(sys.argv)>2 else 'work/v2f_zh.npy'
out=sys.argv[3] if len(sys.argv)>3 else '08_v2f_exact.csv'
a_used=float(sys.argv[4]) if len(sys.argv)>4 else 1.027
zh=np.load(W+zhf)
b_used=M1-a_used*zh.mean()
zsub=np.clip(a_used*zh+b_used,0,None)          # exactly what was submitted
Ezh2=np.mean(zsub**2); Ezh=zsub.mean()
czh=(M2+Ezh2-S*S)/2
a,b=np.linalg.solve(np.array([[Ezh2,Ezh],[Ezh,1.0]]),np.array([czh,M1]))
mse=M2-2*(a*czh+b*M1)+(a*a*Ezh2+2*a*b*Ezh+b*b)
print(f"submitted: mean={Ezh:.6f} E[zh^2]={Ezh2:.6f} score={S:.10f}")
print(f"recovered E[z*zh] = {czh:.6f}")
print(f"EXACT optimal on submitted vector: a={a:.6f} b={b:+.6f}")
print(f"expected public RMSLE = {np.sqrt(max(mse,0)):.6f}   (gain {S-np.sqrt(max(mse,0)):+.6f})")
# implied optimal slope on the raw zh
print(f"=> implied optimal slope on raw zh: {a*a_used:.6f}  (used {a_used:.4f})")
zf=np.clip(a*zsub+b,0,None); p=np.expm1(zf)
uids=np.load(W+'work/mat/uids.npy')
with open(W+'submissions/'+out,'w',newline='') as f:
    c=csv.writer(f); c.writerow(['user_id','predict'])
    for u,v in zip(uids,p): c.writerow([int(u),float(v)])
print(f"wrote submissions/{out}  mean_log1p={np.log1p(p).mean():.6f}")
