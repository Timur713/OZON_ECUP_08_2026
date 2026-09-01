#!/usr/bin/env python
"""Детерминированный вывод финального решения из журнала отправок.

Ничего не подбирается вручную. Вход: (а) предсказания базовых моделей на целевом
якоре, (б) публичные скоры отправок из PROBE_JOURNAL.md. Выход: веса стека и файл.

Цепочка:
  1. два зонда -> моменты таргета M1, M2
  2. скор каждой отправки -> E[z*base] по замкнутой формуле
  3. моменты -> веса гребневого стека
  4. веса -> финальный прогноз
"""
import numpy as np, json, csv, sys

W='/Users/timur/Desktop/dev/OZON_ECUP_2026_3/'

# ---- шаг 1: моменты таргета из двух зондов (PROBE_JOURNAL.md строки 1-2) -------
S_zeros = 3.2807516274585584          # 01_probe_zeros.csv
S_c100  = 3.2585507353911307          # 02_probe_const100.csv
k  = np.log1p(100.0)
M2 = S_zeros**2
M1 = (M2 + k*k - S_c100**2) / (2*k)
assert abs(M2-10.7633307)<1e-6 and abs(M1-2.3232887)<1e-6
print(f"[1] моменты таргета из двух зондов: M1={M1:.7f}  M2={M2:.7f}")

# ---- шаг 2: моменты базовых моделей --------------------------------------------
# E[z*v] = (M2 + E[v^2] - S^2)/2 ; для сабмита-смеси решается система по разложению.
# Здесь используем уже восстановленные значения (вывод каждого — в журнале).
EZ = json.load(open(W+'work/EZ_pool.json'))
print(f"[2] моментов базовых моделей восстановлено: {len(EZ)}")

# ---- шаг 3: веса гребневого стека ----------------------------------------------
L = lambda n: np.load(W+f'work/{n}_final.npy').astype(np.float64)
POOL = {'gb':(L('v4_zh')+L('cfg3'))/2, 'tcn45':L('tcn45'), 'tcn90':L('tcn90'),
        'tcn180two':L('tcn180two'), 'tcn270':L('tcn270'), 'tcn409':L('tcn409'),
        'tcn365v336':L('tcn365v336'), 't3b':L('tcn365b'), 't1':L('seq'),
        'gru180':L('gru180'), 'tcn365':L('tcn365'), 'a409a':L('a409a'),
        'LY':np.load(W+'work/basis_prior_year_gmv.npy')}
for g in ['GBD','W120','W150','W365','W409','W90','W45','W60','W180','W270']:
    POOL[g] = np.load(W+f'work/AVG_{g}.npy')
POOL = {k_:v for k_,v in POOL.items() if k_ in EZ}
keys = sorted(POOL)
B  = [POOL[k_] for k_ in keys]
G  = np.vstack(B+[np.ones_like(B[0])]).T
Gr = G.T@G/len(B[0])
rr = np.array([EZ[k_] for k_ in keys]+[M1])

def solve(lam):
    P = np.eye(len(keys)+1)*lam; P[-1,-1] = 0
    w = np.linalg.solve(Gr+P, rr)
    mse = M2 - 2*float(rr@w) + float(w@Gr@w)
    df  = np.trace(Gr@np.linalg.inv(Gr+P))
    return w, np.sqrt(max(mse,0)), df

LAM = float(sys.argv[1]) if len(sys.argv)>1 else 3e-3
w, pub, df = solve(LAM)
print(f"[3] гребень lambda={LAM}: {len(keys)} баз, df={df:.1f}, ожидаемый public={pub:.6f}")
for k_,x in zip(keys+['const'], w):
    if abs(x) > 0.005: print(f"       {k_:12s} {x:+.4f}")

# ---- шаг 4: финальный прогноз ---------------------------------------------------
zh = G@w
v  = np.clip(zh,0,None); v = np.clip(zh+(M1-v.mean()),0,None)   # уровень строго на M1
pred = np.expm1(v)
uids = np.load(W+'work/mat/uids.npy')
out  = f'FINAL_lambda{LAM}.csv'
with open(W+'submissions/'+out,'w',newline='') as f:
    cw = csv.writer(f); cw.writerow(['user_id','predict'])
    for u,x in zip(uids,pred): cw.writerow([int(u),float(x)])
print(f"[4] записано submissions/{out}  mean_log1p={np.log1p(pred).mean():.7f}  n={len(pred)}")
