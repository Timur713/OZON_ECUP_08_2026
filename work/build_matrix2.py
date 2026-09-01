import polars as pl, numpy as np, os
from datetime import date
W=os.environ.get('ECUP_ROOT','/Users/timur/Desktop/dev/OZON_ECUP_2026_3/').rstrip('/')+'/'; MD=W+'work/mat/'
NEW=['has_search_to_cart','has_search_to_ord','has_cat_to_cart','has_cat_to_ord',
     'search_to_cart','cat_to_cart']
missing=[c for c in NEW if not os.path.exists(MD+c+'.npy')]
print("building:",missing,flush=True)
if missing:
    d=pl.read_parquet(W+'data/train.parquet',columns=['event_date','user_id']+missing)
    D0=date(2025,1,1); NDAY=409
    uids=np.load(MD+'uids.npy'); NU=len(uids)
    umap=pl.DataFrame({'user_id':uids,'ui':np.arange(NU,dtype=np.int32)})
    d=d.join(umap,on='user_id',how='left').with_columns(
        ((pl.col('event_date')-pl.lit(D0)).dt.total_days()).cast(pl.Int32).alias('di'))
    ui=d['ui'].to_numpy(); di=d['di'].to_numpy()
    for c in missing:
        dt=np.int8 if c.startswith('has_') else np.int16
        a=np.zeros((NU,NDAY),dtype=dt); a[ui,di]=d[c].to_numpy().astype(dt)
        np.save(MD+c+'.npy',a)
        print(f"  {c:22s} {dt.__name__:6s} nonzero-days/user={a.astype(bool).sum(1).mean():.1f}",flush=True)
        del a
print("DONE",flush=True)
