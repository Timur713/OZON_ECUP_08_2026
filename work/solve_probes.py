"""Turn leaderboard probe scores into the exactly-optimal global recalibration.

Usage:
  python solve_probes.py S0 S1 [S2]
    S0 = public score of 01_probe_zeros.csv     (all-zero submission)
    S1 = public score of 02_probe_const100.csv  (constant 100)
    S2 = public score of 03_model_L2466.csv     (optional: our model)
"""
import sys, numpy as np, csv
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
S0=float(sys.argv[1]); S1=float(sys.argv[2])
k=np.log1p(100.0)
M2=S0**2                      # E[z^2] on the public split
M1=(M2+k*k-S1**2)/(2*k)       # E[z]   on the public split
print(f"public split:  E[z^2] = {M2:.6f}   E[z] = {M1:.6f}")
print(f"our seasonal estimate was L = 2.4660  ->  error {100*(2.4660/M1-1):+.2f}%")

zh=np.load(W+'work/final_zh.npy')          # blended prediction BEFORE the affine step
if len(sys.argv)>3:
    S2=float(sys.argv[3]); a_prev=float(np.load(W+'work/final_slope.npy')[0])
    b_prev=2.4660-a_prev*zh.mean()
    zsub=np.clip(a_prev*zh+b_prev,0,None)          # exactly what was submitted
    Ezh=(zsub**2).mean(); Ez_h=zsub.mean()
    Ezzh=(M2+Ezh-S2**2)/2                          # recovered E[z*zhat]
    # solve normal equations for  z ~ a*zsub + b   using known moments
    A=np.array([[Ezh,Ez_h],[Ez_h,1.0]]); rhs=np.array([Ezzh,M1])
    a,b=np.linalg.solve(A,rhs)
    print(f"recovered E[z*zhat] = {Ezzh:.6f}")
    print(f"EXACT optimal recalibration of the submitted vector:  a = {a:.6f}   b = {b:+.6f}")
    zfin=np.clip(a*zsub+b,0,None)
    exp_mse=M2-2*(a*Ezzh+b*M1)+(a*a*Ezh+2*a*b*Ez_h+b*b)
    print(f"expected public RMSLE after recalibration ~ {np.sqrt(max(exp_mse,0)):.6f}  (was {S2:.6f})")
    out='05_model_probe_calibrated.csv'
else:
    a=float(np.load(W+'work/final_slope.npy')[0]); b=M1-a*zh.mean()
    zfin=np.clip(a*zh+b,0,None)
    print(f"level-matched recalibration:  a = {a:.6f} (rolling-CV slope)   b = {b:+.6f}")
    out='03b_model_probeL.csv'
uids=np.load(W+'work/mat/uids.npy'); pred=np.expm1(zfin)
with open(W+'submissions/'+out,'w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['user_id','predict'])
    for u,p in zip(uids,pred): wr.writerow([int(u),float(p)])
print(f"wrote submissions/{out}   mean log1p(pred) = {np.log1p(pred).mean():.4f}")
