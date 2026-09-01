import polars as pl, numpy as np, os
from datetime import date, timedelta
D=os.environ.get('ECUP_ROOT','/Users/timur/Desktop/dev/OZON_ECUP_2026_3/').rstrip('/')+'/'
os.makedirs(D+'work/mat',exist_ok=True)
d = pl.read_parquet(D+'data/train.parquet')
D0 = date(2025,1,1); D1 = date(2026,2,13); NDAY=(D1-D0).days+1
uids = d['user_id'].unique().sort().to_numpy(); NU=len(uids)
print("users",NU,"days",NDAY)
np.save(D+'work/mat/uids.npy',uids)
umap = pl.DataFrame({'user_id':uids,'ui':np.arange(NU,dtype=np.int32)})
d = d.join(umap,on='user_id',how='left').with_columns(
        ((pl.col('event_date')-pl.lit(D0)).dt.total_days()).cast(pl.Int32).alias('di'))
ui = d['ui'].to_numpy(); di = d['di'].to_numpy()
specs = [('gmv',np.float32),('gmv_search',np.float32),('gmv_cat',np.float32),
         ('to_ord',np.int16),('to_cart',np.int16),('searches',np.int16),
         ('search',np.int8),('cat',np.int8),
         ('search_to_ord',np.int16),('cat_to_ord',np.int16)]
for col,dt in specs:
    a = np.zeros((NU,NDAY),dtype=dt)
    v = d[col].to_numpy()
    a[ui,di] = v.astype(dt)
    np.save(D+f'work/mat/{col}.npy',a)
    print("saved",col,dt,a.nbytes/1e6,"MB")
    del a
act = np.zeros((NU,NDAY),dtype=np.int8); act[ui,di]=1
np.save(D+'work/mat/active.npy',act)
print("done")
