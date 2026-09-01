#!/usr/bin/env python
"""ГЛАВНАЯ СТАВКА: мультизадачный классификатор.

Меняет одновременно ПОСТАНОВКУ и ЁМКОСТЬ — единственная непроверенная комбинация.
82% достижимой дисперсии таргета — это «купит или нет», поэтому основная ёмкость
уходит в бинарные головы, а не в регрессор.

  головы: P(y>0) на горизонтах 7/14/30/60 дней + условная величина E[z|y>0]
  выход:  zhat = P(y>0 за 30д) * E[z|y>0]
  масштаб: --width 384 --blocks 12  (против 96/8 у текущих сетей)

Запуск:
  python train_classifier.py <tag> --window 409 --width 384 --blocks 12 \
      --epochs 3 --seed 1 --stride 4 --frac 0.25 --channels all --bs 2048
"""
import numpy as np, torch, torch.nn as nn, time, gc, argparse, os, json
MD=os.environ.get('ECUP_MAT','/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/')
OUT=os.environ.get('ECUP_OUT','/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/')

def pick_device():
    if torch.cuda.is_available(): return 'cuda'
    if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

CHSET={'base':['gmv','to_ord','to_cart','searches','active','search','cat'],
       'all':['gmv','to_ord','to_cart','searches','active','search','cat',
              'gmv_search','gmv_cat','search_to_ord','cat_to_ord',
              'has_search_to_ord','has_search_to_cart','has_cat_to_ord','has_cat_to_cart',
              'search_to_cart','cat_to_cart']}
HOR=[7,14,30,60]                      # multi-task horizons; 30 is the real target

class Net(nn.Module):
    def __init__(s,C,width,blocks,L):
        super().__init__()
        s.inp=nn.Conv1d(C,width,5,padding=2)
        s.blocks=nn.ModuleList()
        d=1
        for _ in range(blocks):
            s.blocks.append(nn.Sequential(nn.Conv1d(width,width,3,padding=d,dilation=d),
                                          nn.GELU(),nn.BatchNorm1d(width)))
            d=min(d*2,L//4)
        h=width*3
        s.trunk=nn.Sequential(nn.Linear(h,512),nn.GELU(),nn.Dropout(0.1),nn.Linear(512,256),nn.GELU())
        s.cls=nn.Linear(256,len(HOR))     # P(y>0) per horizon
        s.val=nn.Linear(256,1)            # E[z | y>0] for the 30d horizon
    def forward(s,x):
        y=torch.relu(s.inp(x))
        for b in s.blocks: y=y+b(y)
        f=torch.cat([y.mean(-1),y.max(-1).values,y[...,-14:].mean(-1)],1)
        t=s.trunk(f)
        return s.cls(t), s.val(t).squeeze(-1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('tag'); p.add_argument('--window',type=int,default=409)
    p.add_argument('--width',type=int,default=384); p.add_argument('--blocks',type=int,default=12)
    p.add_argument('--epochs',type=int,default=3); p.add_argument('--seed',type=int,default=1)
    p.add_argument('--stride',type=int,default=4); p.add_argument('--frac',type=float,default=0.25)
    p.add_argument('--channels',default='all'); p.add_argument('--bs',type=int,default=2048)
    p.add_argument('--lr',type=float,default=2e-3); p.add_argument('--val',type=int,default=378)
    p.add_argument('--nusers',type=int,default=0, help='toy mode: limit users')
    a=p.parse_args()
    dev=pick_device(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    CH=CHSET[a.channels]
    mats=[]
    for c in CH:
        x=np.load(MD+c+'.npy')
        if a.nusers: x=x[:a.nusers]
        x=x.astype(np.float32)
        if c!='active' and not c.startswith('has_'): np.log1p(x,out=x)
        mats.append(x.astype(np.float16))
    M=np.stack(mats,1); del mats; gc.collect()
    NU,C,ND=M.shape; L=a.window; FINAL=ND-1
    sd=(M[::37].astype(np.float32).reshape(-1,ND).std()+1e-6).astype(np.float32)
    gmv=np.load(MD+'gmv.npy')[:NU]
    cs=np.zeros((NU,ND+1),dtype=np.float64); np.cumsum(gmv,axis=1,out=cs[:,1:]); del gmv
    TR=[t for t in range(60,a.val-max(HOR),a.stride)]
    def tgt(t,h):
        b=min(ND-1,t+h); return cs[:,b+1]-cs[:,t+1]
    Y={t:{h:np.log1p(tgt(t,h)).astype(np.float32) for h in HOR} for t in TR+[a.val]}
    print(f"{a.tag}: dev={dev} L={L} width={a.width} blocks={a.blocks} ch={C} "
          f"anchors={len(TR)} users={NU} M={M.nbytes/1e9:.2f}GB",flush=True)
    net=Net(C,a.width,a.blocks,L).to(dev)
    npar=sum(p_.numel() for p_ in net.parameters())
    print(f"  parameters: {npar/1e6:.2f}M",flush=True)
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
    steps=a.epochs*len(TR)*(int(NU*a.frac)//a.bs)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=max(steps,1),pct_start=0.15)
    bce=nn.BCEWithLogitsLoss()
    AMP = (dev=='cuda')                      # mixed precision: 2-3x on tensor cores
    scaler=torch.amp.GradScaler('cuda',enabled=AMP)
    if AMP: torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=True
    def win(idx,t):
        lo=t-L+1
        if lo>=0: return M[idx,:,lo:t+1].astype(np.float32)
        o=np.zeros((len(idx),C,L),dtype=np.float32); o[:,:,-(t+1):]=M[idx,:,:t+1].astype(np.float32); return o
    def predict(t):
        net.eval(); out=np.empty(NU,dtype=np.float32)
        with torch.no_grad():
            for i in range(0,NU,4096):
                ii=np.arange(i,min(i+4096,NU))
                with torch.amp.autocast('cuda',enabled=(dev=='cuda')):
                    lg,v=net(torch.from_numpy(win(ii,t)/sd).to(dev))
                p30=torch.sigmoid(lg[:,HOR.index(30)])
                out[ii]=(p30*v).float().cpu().numpy()
        net.train(); return out
    def rmsle(z,x): return float(np.sqrt(np.mean((z-np.clip(x,0,None))**2)))
    def cal(zh,z):
        A=np.vstack([zh,np.ones_like(zh)]).T; c,_,_,_=np.linalg.lstsq(A,z,rcond=None); return rmsle(z,A@c)
    rng=np.random.default_rng(a.seed); t0=time.time(); step=0; best=(9,None,None)
    for ep in range(a.epochs):
        for t in rng.permutation(TR):
            perm=rng.permutation(NU)[:int(NU*a.frac)]
            for i in range(0,len(perm)-a.bs+1,a.bs):
                idx=np.sort(perm[i:i+a.bs])
                x=torch.from_numpy(win(idx,t)/sd).to(dev,non_blocking=True)
                with torch.amp.autocast('cuda',enabled=AMP):
                    lg,v=net(x)
                    loss=0.
                    for hi,h in enumerate(HOR):
                        yb=torch.from_numpy((Y[t][h][idx]>0).astype(np.float32)).to(dev)
                        loss=loss+bce(lg[:,hi],yb)
                    y30=torch.from_numpy(Y[t][30][idx]).to(dev)
                    p30=torch.sigmoid(lg[:,HOR.index(30)])
                    loss=loss+3.0*nn.functional.mse_loss(p30*v,y30)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); step+=1
                if step%500==0: print(f"  ep{ep} {step}/{steps} loss={loss.item():.4f} {time.time()-t0:.0f}s",flush=True)
        pv=predict(a.val); r=cal(pv,Y[a.val][30])
        print(f"EPOCH {ep}: val cal = {r:.5f} ({time.time()-t0:.0f}s)",flush=True)
        if r<best[0]:
            best=(r,pv.copy(),predict(FINAL).copy())
            np.save(OUT+f'{a.tag}_val.npy',best[1]); np.save(OUT+f'{a.tag}_final.npy',best[2])
            print("   ^ new best, saved",flush=True)
    print(f"BEST {a.tag}: {best[0]:.5f}",flush=True)
if __name__=='__main__': main()
