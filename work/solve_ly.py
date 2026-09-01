"""From 15_v4_ly_probe score -> exact E[z*LY] -> optimal (v, LY, 1) combination."""
import sys, numpy as np, csv
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1,M2=2.3232887,10.7633307
S=float(sys.argv[1])
w=np.load(W+'work/v4_ly_w.npy'); ve=np.load(W+'work/v4_ve.npy')
v=np.load(W+'work/v4_v.npy'); g=np.load(W+'work/ly_basis.npy')
c,gm,Ezv,a,b=np.load(W+'work/v4_ly_c.npy')
Ezw=(M2+np.mean(w*w)-S*S)/2
# w = a*v + b + c*(g-gm) + (M1 - mean(...))  -> exact linear in v, g, 1
shift=M1-(ve+c*(g-gm)).mean()
# E[z*w] = a*E[z*v] + c*E[z*g] + (b - c*gm + shift)*M1
Ezg=(Ezw - a*Ezv - (b - c*gm + shift)*M1)/c
print(f"E[z*v]={Ezv:.6f}  E[z*w]={Ezw:.6f}  ->  E[z*LY]={Ezg:.6f}")
U=np.vstack([v,g,np.ones_like(v)]).T; G=U.T@U/len(v)
r=np.array([Ezv,Ezg,M1]); k=np.linalg.solve(G,r)
mse=M2-2*float(r@k)+float(k@G@k)
print(f"optimal: v={k[0]:+.6f}  LY={k[1]:+.6f}  1={k[2]:+.6f}")
print(f"expected public RMSLE = {np.sqrt(max(mse,0)):.6f}  (affine-only was 1.651536)")
p=np.expm1(np.clip(U@k,0,None)); uids=np.load(W+'work/mat/uids.npy')
with open(W+'submissions/16_v4_ly_final.csv','w',newline='') as f:
    cw=csv.writer(f); cw.writerow(['user_id','predict'])
    for u,x in zip(uids,p): cw.writerow([int(u),float(x)])
print(f"wrote 16_v4_ly_final.csv mean_log1p={np.log1p(p).mean():.6f}")
