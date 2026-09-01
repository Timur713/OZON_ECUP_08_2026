import polars as pl, numpy as np
from datetime import date, timedelta
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
daily = pl.read_parquet('/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/daily.parquet')

def win(a,b):
    s = daily.filter(pl.col("event_date").is_between(date.fromisoformat(a),date.fromisoformat(b)))
    return dict(gmv=s["gmv"].sum(), buyers=s["n_buyers"].sum(), ords=s["ord"].sum(), act=s["n_active"].sum(), days=len(s))

print("=== 2025 analogue windows ===")
w_tgt25 = win("2025-02-14","2025-03-15")   # same calendar window, 1 yr earlier
w_nai25 = win("2025-01-15","2025-02-13")   # naive window analogue
print("target-window 2025 :", w_tgt25)
print("naive-window  2025 :", w_nai25)
print("RATIO gmv tgt/naive (2025):", w_tgt25['gmv']/w_nai25['gmv'])
print("RATIO ord tgt/naive (2025):", w_tgt25['ords']/w_nai25['ords'])
print("RATIO buyerdays     (2025):", w_tgt25['buyers']/w_nai25['buyers'])

print("\n=== 2026 naive window (this is what sample_submit contains) ===")
w_nai26 = win("2026-01-15","2026-02-13")
print("naive-window 2026 :", w_nai26)
print("YoY naive window 26/25:", w_nai26['gmv']/w_nai25['gmv'])

print("\n=== daily series Feb 1 - Mar 20, 2025 (holiday shape) ===")
print(daily.filter(pl.col("event_date").is_between(date(2025,2,1),date(2025,3,20)))
      .with_columns((pl.col("gmv")/1000).round(1).alias("gmv_k"),
                    pl.col("event_date").dt.strftime("%a").alias("dow"))
      .select("event_date","dow","gmv_k","n_buyers","ord"))

print("\n=== daily series Jan 2026 - Feb 13 2026 (tail of train) ===")
print(daily.filter(pl.col("event_date")>=date(2026,1,20))
      .with_columns((pl.col("gmv")/1000).round(1).alias("gmv_k"),
                    pl.col("event_date").dt.strftime("%a").alias("dow"))
      .select("event_date","dow","gmv_k","n_buyers","ord"))
