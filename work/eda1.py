import polars as pl, numpy as np
from datetime import date, timedelta

d = pl.read_parquet('/Users/timur/Desktop/dev/OZON_ECUP_2026_3/data/train.parquet')
print("shape", d.shape)
print(d.schema)
print(d.head(3))

daily = (d.group_by("event_date").agg(
    pl.len().alias("n_active"),
    pl.sum("gmv").alias("gmv"),
    pl.sum("to_ord").alias("ord"),
    pl.sum("to_cart").alias("cart"),
    pl.sum("searches").alias("srch"),
    (pl.col("gmv")>0).sum().alias("n_buyers"),
).sort("event_date"))
daily.write_parquet("/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/daily.parquet")

# monthly seasonal index
m = (daily.with_columns(pl.col("event_date").dt.strftime("%Y-%m").alias("ym"))
     .group_by("ym").agg(pl.mean("gmv").alias("gmv_per_day"),
                         pl.mean("n_buyers").alias("buyers_per_day"),
                         pl.mean("n_active").alias("active_per_day"),
                         pl.mean("ord").alias("ord_per_day")).sort("ym"))
print("\n=== MONTHLY (per-day averages) ===")
print(m)
