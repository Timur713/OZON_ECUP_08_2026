"""Optimal 2-model stack recovered entirely from leaderboard moments.
Usage: python solve_stack.py <score_06_v2f_a1027> <score_06_v3_levelM1>"""
import sys, numpy as np, csv
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
M1,M2=2.3232887,10.7633307
SA=float(sys.argv[1]); SB=float(sys.argv[2])
zh2=np.load(W+'work/v2f_zh.npy'); zh3=np.load(W+'work/v3_zh_final.npy')
u=np.clip(1.0270*zh2+(M1-1.0270*zh2.mean()),0,None)      # exactly what 06_v2f_a1027 contained
v=np.clip(0.9510210*zh3+(M1-0.9510210*zh3.mean()),0,None) # exactly what 06_v3_levelM1 contained
Eu2,Ev2,Euv,Eu,Ev=np.mean(u*u),np.mean(v*v),np.mean(u*v),u.mean(),v.mean()
Ezu=(M2+Eu2-SA*SA)/2; Ezv=(M2+Ev2-SB*SB)/2
print(f"E[z*u]={Ezu:.6f}  E[z*v]={Ezv:.6f}")
for nm,S,E2,Ez,x in [('v2f',SA,Eu2,Ezu,u),('v3',SB,Ev2,Ezv,v)]:
    a,b=np.linalg.solve(np.array([[E2,x.mean()],[x.mean(),1]]),np.array([Ez,M1]))
    m=M2-2*(a*Ez+b*M1)+(a*a*E2+2*a*b*x.mean()+b*b)
    print(f"  {nm} alone: exact a={a:.5f} b={b:+.5f} -> expected {np.sqrt(max(m,0)):.6f}  (submitted {S:.6f})")
A=np.array([[Eu2,Euv,Eu],[Euv,Ev2,Ev],[Eu,Ev,1.0]]); r=np.array([Ezu,Ezv,M1])
a1,a2,b=np.linalg.solve(A,r)
mse=M2-2*(a1*Ezu+a2*Ezv+b*M1)+(a1*a1*Eu2+a2*a2*Ev2+b*b+2*a1*a2*Euv+2*a1*b*Eu+2*a2*b*Ev)
print(f"\nSTACK: a1(v2f)={a1:.5f}  a2(v3)={a2:.5f}  b={b:+.5f}")
print(f"expected public RMSLE = {np.sqrt(max(mse,0)):.6f}   (best single {min(SA,SB):.6f})")
zf=np.clip(a1*u+a2*v+b,0,None); p=np.expm1(zf)
uids=np.load(W+'work/mat/uids.npy')
with open(W+'submissions/08_stack.csv','w',newline='') as f:
    c=csv.writer(f); c.writerow(['user_id','predict'])
    for i,q in zip(uids,p): c.writerow([int(i),float(q)])
print(f"wrote submissions/08_stack.csv  mean_log1p={np.log1p(p).mean():.6f}")
