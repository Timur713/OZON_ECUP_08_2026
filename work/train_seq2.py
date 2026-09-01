"""Sequence models over raw daily series.
Usage: train_seq2.py <tag> <arch:tcn|gru> <L> <epochs> <seed> <head:direct|two>"""
import numpy as np, torch, torch.nn as nn, time, gc, sys, os
W=os.environ.get('ECUP_ROOT','/Users/timur/Desktop/dev/OZON_ECUP_2026_3/').rstrip('/')+'/'
MD=os.environ.get('ECUP_MAT',W+'work/mat/').rstrip('/')+'/'
TAG,ARCH,L,EP,SEED,HEAD=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),sys.argv[6]
STRIDE=int(sys.argv[7]) if len(sys.argv)>7 else 12
VALA=int(sys.argv[8]) if len(sys.argv)>8 else 378
FRAC=float(sys.argv[10]) if len(sys.argv)>10 else 1.0
def pick_device():
    if torch.cuda.is_available(): return 'cuda'
    if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'
dev=pick_device()
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"{TAG}: arch={ARCH} L={L} ep={EP} seed={SEED} head={HEAD} stride={STRIDE} dev={dev}",flush=True)
VAL=VALA; FINAL=408
TR=[t for t in range(60,VAL-29,STRIDE)] if STRIDE<12 else [t for t in range(186,VAL-29,STRIDE)]
CHSET={'base':['gmv','to_ord','to_cart','searches','active','search','cat'],
       'full':['gmv','to_ord','to_cart','searches','active','search','cat',
               'gmv_search','gmv_cat','search_to_ord','cat_to_ord'],
       'all':['gmv','to_ord','to_cart','searches','active','search','cat',
              'gmv_search','gmv_cat','search_to_ord','cat_to_ord',
              'has_search_to_ord','has_search_to_cart','has_cat_to_ord','has_cat_to_cart',
              'search_to_cart','cat_to_cart']}
CH=CHSET[sys.argv[9] if len(sys.argv)>9 else 'base']
mats=[]
for c in CH:
    a=np.load(MD+c+'.npy').astype(np.float32)
    if c!='active' and not c.startswith('has_'): np.log1p(a,out=a)
    mats.append(a.astype(np.float16))          # fp16 storage: 17ch fits in 3.5GB not 7GB
M=np.stack(mats,1); del mats; gc.collect()
NU,C,ND=M.shape
sd=(M[::37].astype(np.float32).reshape(-1,ND).std()+1e-6).astype(np.float32)
gmv=np.load(MD+'gmv.npy'); cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:]); del gmv
Z={t:np.log1p(cs[:,min(ND-1,t+30)+1]-cs[:,t+1]).astype(np.float32) for t in TR+[VAL]}
B={t:(Z[t]>0).astype(np.float32) for t in TR}
BS=1024
print(f"anchors={len(TR)} frac={FRAC} steps/epoch={len(TR)*(int(NU*FRAC)//BS)} M={M.shape} {M.nbytes/1e9:.2f}GB",flush=True)

class TCN(nn.Module):
    def __init__(s,C,h=96,nout=1):
        super().__init__(); s.inp=nn.Conv1d(C,h,5,padding=2); s.blocks=nn.ModuleList()
        for d in (1,2,4,8,16,32,64,128):
            if d*2>L: break
            s.blocks.append(nn.Sequential(nn.Conv1d(h,h,3,padding=d,dilation=d),nn.GELU(),nn.BatchNorm1d(h)))
        s.head=nn.Sequential(nn.Linear(h*3,128),nn.GELU(),nn.Linear(128,nout))
    def forward(s,x):
        y=torch.relu(s.inp(x))
        for b in s.blocks: y=y+b(y)
        return s.head(torch.cat([y.mean(-1),y.max(-1).values,y[...,-14:].mean(-1)],1))
class GRU(nn.Module):
    def __init__(s,C,h=112,nout=1):
        super().__init__(); s.g=nn.GRU(C,h,num_layers=2,batch_first=True,dropout=0.1)
        s.head=nn.Sequential(nn.Linear(h*2,128),nn.GELU(),nn.Linear(128,nout))
    def forward(s,x):
        y,_=s.g(x.transpose(1,2))
        return s.head(torch.cat([y[:,-1],y.mean(1)],1))
nout=2 if HEAD=='two' else 1
net=(TCN(C,nout=nout) if ARCH=='tcn' else GRU(C,nout=nout)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4)
steps=EP*len(TR)*(int(NU*FRAC)//BS)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=steps,pct_start=0.15)
def rmsle(z,p): return float(np.sqrt(np.mean((z-np.clip(p,0,None))**2)))
def cal(zh,z):
    A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None); return rmsle(z,A@c)
def win(idx,t):
    lo=t-L+1
    if lo>=0: return M[idx,:,lo:t+1].astype(np.float32)
    o=np.zeros((len(idx),C,L),dtype=np.float32); o[:,:,-(t+1):]=M[idx,:,:t+1].astype(np.float32); return o
def predict(t):
    net.eval(); out=np.empty(NU,dtype=np.float32)
    with torch.no_grad():
        for i in range(0,NU,4096):
            ii=np.arange(i,min(i+4096,NU))
            o=net(torch.from_numpy(win(ii,t)/sd).to(dev))
            out[ii]=(torch.sigmoid(o[:,0])*o[:,1] if nout==2 else o[:,0]).float().cpu().numpy()
    net.train(); return out
t0=time.time(); rng=np.random.default_rng(SEED); step=0; best=(9,None,None)
bce=nn.BCEWithLogitsLoss()
for ep in range(EP):
    for t in rng.permutation(TR):
        z=Z[t]; bb=B[t]; perm=rng.permutation(NU)
        NUSE=int(NU*FRAC)                      # subsample users per anchor:
        perm=perm[:NUSE]                       # more anchors at the SAME step budget
        for i in range(0,NUSE-BS+1,BS):
            idx=np.sort(perm[i:i+BS])
            x=torch.from_numpy(win(idx,t)/sd).to(dev); y=torch.from_numpy(z[idx]).to(dev)
            o=net(x)
            if nout==2:
                yb=torch.from_numpy(bb[idx]).to(dev)
                loss=bce(o[:,0],yb)+nn.functional.mse_loss(torch.sigmoid(o[:,0])*o[:,1],y)
            else: loss=nn.functional.mse_loss(o[:,0],y)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); step+=1
            if step%600==0: print(f"  ep{ep} {step}/{steps} loss={loss.item():.4f} {time.time()-t0:.0f}s",flush=True)
    pv=predict(VAL); r=cal(pv,Z[VAL])
    print(f"EPOCH {ep}: val cal = {r:.5f}  ({time.time()-t0:.0f}s)",flush=True)
    if r<best[0]:
        best=(r,pv.copy(),predict(FINAL).copy())
        np.save(W+f'work/{TAG}_val.npy',best[1]); np.save(W+f'work/{TAG}_final.npy',best[2])
        print(f"   ^ new best, saved",flush=True)
print(f"BEST {TAG}: {best[0]:.5f}",flush=True)
