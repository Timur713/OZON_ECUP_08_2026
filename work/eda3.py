import polars as pl, numpy as np
from datetime import date, timedelta
D='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'
d = pl.read_parquet(D+'data/train.parquet', columns=['event_date','user_id','gmv','to_ord','searches'])
users = pl.DataFrame({'user_id': d['user_id'].unique().sort()})
print("n_users", len(users))

def wsum(a,b,col='gmv',name='v'):
    return (d.filter(pl.col('event_date').is_between(a,b))
             .group_by('user_id').agg(pl.col(col).sum().alias(name)))

def rmsle(y,p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(y,0,None))-np.log1p(np.clip(p,0,None)))**2)))

# --- 1) verify sample_submit == GMV(2026-01-15..2026-02-13)
ss = pl.read_csv(D+'example/sample_submit.csv')
nai26 = users.join(wsum(date(2026,1,15),date(2026,2,13),name='v'),on='user_id',how='left').with_columns(pl.col(pl.Float64).fill_null(0.0))
chk = ss.join(nai26,on='user_id',how='inner')
print("sample_submit == naive Jan15-Feb13 2026 ?  max abs diff =",
      float((chk['predict']-chk['v']).abs().max()), " n=",len(chk))

# --- 2) 2025 analogue: features window = Jan15-Feb13 2025, target = Feb14-Mar15 2025
prev25 = users.join(wsum(date(2025,1,15),date(2025,2,13),name='prev'),on='user_id',how='left').with_columns(pl.col(pl.Float64).fill_null(0.0))
tgt25  = users.join(wsum(date(2025,2,14),date(2025,3,15),name='y'),   on='user_id',how='left').with_columns(pl.col(pl.Float64).fill_null(0.0))
df = prev25.join(tgt25,on='user_id')
prev = df['prev'].to_numpy(); y = df['y'].to_numpy()

# restrict to users that existed by then (had ANY activity before 2025-02-14)
seen = (d.filter(pl.col('event_date')<date(2025,2,14)).select('user_id').unique()
          .with_columns(pl.lit(1).alias('seen')))
df2 = df.join(seen,on='user_id',how='left').fill_null(0)
mask = df2['seen'].to_numpy().astype(bool)
print("\nusers active before 2025-02-14:", mask.sum(), "of", len(mask))

print("\n=== target stats (2025 analogue, ALL 250k users) ===")
print("share y>0 :", float((y>0).mean()), " | mean log1p(y):", float(np.log1p(y).mean()),
      " | E[log1p^2]:", float((np.log1p(y)**2).mean()))
print("share prev>0:", float((prev>0).mean()))

print("\n=== RMSLE of naive autoregression, and of scaled variants ===")
print(f"{'multiplier':>12} {'RMSLE(all)':>12} {'RMSLE(seen)':>12}")
for mlt in [1.0,1.05,1.10,1.163,1.20,1.25,1.30,1.4,1.5]:
    print(f"{mlt:12.3f} {rmsle(y,prev*mlt):12.6f} {rmsle(y[mask],prev[mask]*mlt):12.6f}")

# optimal affine in log space: z_hat = a*log1p(prev)+b
X = np.log1p(prev); Z = np.log1p(y)
A = np.vstack([X,np.ones_like(X)]).T
coef,_,_,_ = np.linalg.lstsq(A,Z,rcond=None)
a,b = coef
print(f"\noptimal affine in log-space: a={a:.4f} b={b:.4f} -> RMSLE={rmsle(y,np.expm1(a*X+b)):.6f}")
# constrained b=0
a0 = float((X@Z)/(X@X))
print(f"pure slope (b=0): a={a0:.4f} -> RMSLE={rmsle(y,np.expm1(a0*X)):.6f}")

# best constant prediction
zc = Z.mean()
print(f"\nbest constant: log1p(c)={zc:.4f} c={np.expm1(zc):.3f} -> RMSLE={rmsle(y,np.full_like(y,np.expm1(zc))):.6f}")
print(f"all-zeros                       -> RMSLE={rmsle(y,np.zeros_like(y)):.6f}")
