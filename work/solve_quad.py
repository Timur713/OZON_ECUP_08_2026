"""From the score of 11_v3_quad_exact.csv recover E[z*v^2] and emit the exact 3-param optimum."""
import sys, numpy as np, csv
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1,M2=2.3232887,10.7633307; S_v3=1.6579427754768816
S=float(sys.argv[1])
zh3=np.load(W+'work/v3_zh_final.npy')
v=np.clip(0.9510210*zh3+(M1-0.9510210*zh3.mean()),0,None)
w=np.load(W+'work/quad11_w.npy'); c2,c1,c0=np.load(W+'work/quad11_coef.npy')
Ezw=(M2+np.mean(w*w)-S*S)/2
Ezv=(M2+np.mean(v*v)-S_v3**2)/2
Ezv2=(Ezw-c1*Ezv-c0*M1)/c2                      # exact, w is an exact quadratic in v
print(f"E[z*v]={Ezv:.6f}  E[z*w]={Ezw:.6f}  ->  E[z*v^2]={Ezv2:.6f}")
u=np.vstack([v*v,v,np.ones_like(v)]).T
G=u.T@u/len(v); r=np.array([Ezv2,Ezv,M1])
a2,a1,a0=np.linalg.solve(G,r)
zf=a2*v*v+a1*v+a0
mse=M2-2*(a2*Ezv2+a1*Ezv+a0*M1)+float(np.array([a2,a1,a0])@G@np.array([a2,a1,a0]))
print(f"EXACT quadratic optimum: {a2:+.6f} v^2 {a1:+.6f} v {a0:+.6f}")
print(f"expected public RMSLE = {np.sqrt(max(mse,0)):.6f}   (11 scored {S:.6f})")
zf=np.clip(zf,0,None); p=np.expm1(zf); uids=np.load(W+'work/mat/uids.npy')
with open(W+'submissions/12_v3_quad_final.csv','w',newline='') as f:
    cw=csv.writer(f); cw.writerow(['user_id','predict'])
    for i,x in zip(uids,p): cw.writerow([int(i),float(x)])
print(f"wrote 12_v3_quad_final.csv  mean_log1p={np.log1p(p).mean():.6f}")
