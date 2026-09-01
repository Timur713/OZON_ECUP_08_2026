import numpy as np, torch, torch.nn as nn, time, gc, sys
MD='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/'
W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
dev='mps' if torch.backends.mps.is_available() else 'cpu'
print("device",dev,flush=True)
L=180; VAL=378; FINAL=408
TR=[t for t in range(186,VAL-29,12)]
CH=['gmv','to_ord','to_cart','searches','active','search','cat']
mats=[]
for c in CH:
    a=np.load(MD+c+'.npy').astype(np.float32)
    if c in ('gmv',): np.log1p(a,out=a)
    elif c in ('to_ord','to_cart','searches'): np.log1p(a,out=a)
    mats.append(a)
M=np.stack(mats,1)              # (NU, C, ND)
del mats; gc.collect()
NU,C,ND=M.shape; print("M",M.shape,M.nbytes/1e9,"GB",flush=True)
sd=M.reshape(NU*C,-1).std()+1e-6
gmv=np.load(MD+'gmv.npy'); cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:]); del gmv
def tgt(t): return cs[:,min(ND-1,t+30)+1]-cs[:,t+1]
Z={t:np.log1p(tgt(t)).astype(np.float32) for t in TR+[VAL]}

class TCN(nn.Module):
    def __init__(s,C,h=96):
        super().__init__()
        s.inp=nn.Conv1d(C,h,5,padding=2)
        s.blocks=nn.ModuleList()
        for d in (1,2,4,8,16,32,64):
            s.blocks.append(nn.Sequential(nn.Conv1d(h,h,3,padding=d,dilation=d),
                                          nn.GELU(),nn.BatchNorm1d(h)))
        s.head=nn.Sequential(nn.Linear(h*3,128),nn.GELU(),nn.Linear(128,1))
    def forward(s,x):
        y=torch.relu(s.inp(x))
        for b in s.blocks: y=y+b(y)
        f=torch.cat([y.mean(-1),y.max(-1).values,y[...,-14:].mean(-1)],1)
        return s.head(f).squeeze(-1)

net=TCN(C).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
BS=1024; EP=int(sys.argv[1]) if len(sys.argv)>1 else 4
steps=EP*len(TR)*(NU//BS)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=steps)
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None)
    return rmsle(z,A@c)
def win(idx,t):
    lo=t-L+1
    if lo>=0: return M[idx,:,lo:t+1]
    out=np.zeros((len(idx),C,L),dtype=np.float32); out[:,:,-(t+1):]=M[idx,:,:t+1]; return out
def predict(t):
    net.eval(); out=np.empty(NU,dtype=np.float32)
    with torch.no_grad():
        for i in range(0,NU,4096):
            x=torch.from_numpy(win(np.arange(i,min(i+4096,NU)),t)/sd).to(dev)
            out[i:i+4096]=net(x).float().cpu().numpy()
    net.train(); return out
t0=time.time(); rng=np.random.default_rng(0); step=0
for ep in range(EP):
    for t in rng.permutation(TR):
        z=Z[t]; perm=rng.permutation(NU)
        for i in range(0,NU-BS+1,BS):
            idx=np.sort(perm[i:i+BS])
            x=torch.from_numpy(win(idx,t)/sd).to(dev)
            y=torch.from_numpy(z[idx]).to(dev)
            loss=nn.functional.mse_loss(net(x),y)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); step+=1
            if step%400==0: print(f"  ep{ep} step{step}/{steps} loss={loss.item():.4f} {time.time()-t0:.0f}s",flush=True)
    pv=predict(VAL); print(f"EPOCH {ep}: val cal RMSLE = {cal(pv,Z[VAL]):.5f}  ({time.time()-t0:.0f}s)",flush=True)
    np.save(W+'work/seq_val.npy',pv)
    torch.save(net.state_dict(),W+'work/seq_net.pt')
pf=predict(FINAL); np.save(W+'work/seq_final.npy',pf)
print("DONE",flush=True)
