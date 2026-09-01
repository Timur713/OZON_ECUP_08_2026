#!/usr/bin/env python
"""Scaled hurdle GBDT: anchor density, learning rate and rounds as knobs.

Hunts a coarse base-level gain on untouched fold 378.  The hurdle split is
exact because z = log1p(y) is exactly 0 when y = 0, so
E[z|x] = P(y>0|x) * E[z|x, y>0].
"""
import argparse, gc, json, os, sys, time
import numpy as np

ROOT = "/home/ubuntu/ecup"
sys.path.insert(0, ROOT + "/work")
import feats4, lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("--anchor-stride", type=int, default=12)
parser.add_argument("--learning-rate", type=float, default=0.05)
parser.add_argument("--rounds", type=int, default=0, help="0 = scale 250 by anchor count/14")
parser.add_argument("--leaves", type=int, default=127)
parser.add_argument("--seeds", type=int, default=2)
parser.add_argument("--direct-weight", type=float, default=0.3)
args = parser.parse_args()

VAL, FINAL = 378, 408
TR = list(range(186, 349, args.anchor_stride))
rounds = args.rounds or int(250 * len(TR) / 14.0)
base = dict(learning_rate=args.learning_rate, num_leaves=args.leaves,
            min_data_in_leaf=200, feature_fraction=0.6, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=10.0, num_threads=32, verbose=-1, max_bin=255)

started = time.time()
X, names = feats4.build(TR)
y = feats4.targets(TR); z = np.log1p(y); b = (y > 0).astype(np.int8); pos = b == 1
print(f"{args.tag}: anchors={len(TR)} X={X.shape} {X.nbytes/1e9:.1f}GB rounds={rounds} "
      f"lr={args.learning_rate} leaves={args.leaves} buyers={pos.mean():.4f} "
      f"{time.time()-started:.0f}s", flush=True)

Xv, _ = feats4.build([VAL]); yv = feats4.targets([VAL]); zv = np.log1p(yv)

direct, hurdle = [], []
for seed in range(1, args.seeds + 1):
    P = {**base, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
    md = lgb.train({**P, "objective": "regression"},
                   lgb.Dataset(X, z, feature_name=names), num_boost_round=rounds)
    mc = lgb.train({**P, "objective": "binary"},
                   lgb.Dataset(X, b, feature_name=names), num_boost_round=rounds)
    mr = lgb.train({**P, "objective": "regression"},
                   lgb.Dataset(X[pos], z[pos], feature_name=names), num_boost_round=rounds)
    direct.append(md.predict(Xv))
    hurdle.append(mc.predict(Xv) * mr.predict(Xv))
    print(f"  seed{seed} done {time.time()-started:.0f}s", flush=True)
    for model, kind in ((md, "direct"), (mc, "class"), (mr, "positive")):
        model.save_model(f"{ROOT}/work/{args.tag}_s{seed}_{kind}.txt")
del X; gc.collect()

prediction = args.direct_weight * np.mean(direct, 0) + (1 - args.direct_weight) * np.mean(hurdle, 0)

rng = np.random.default_rng(20260825)
nusers = len(zv)
calib = np.sort(rng.choice(nusers, nusers // 5, replace=False))
mask = np.ones(nusers, bool); mask[calib] = False
score_idx = np.flatnonzero(mask)
design = np.column_stack([prediction, np.ones(nusers)])
coef = np.linalg.lstsq(design[calib], zv[calib], rcond=None)[0]
fitted = np.clip(design[score_idx] @ coef, 0, None)
score = float(np.sqrt(np.mean((zv[score_idx] - fitted) ** 2)))

np.save(f"{ROOT}/work/{args.tag}_val.npy", prediction.astype(np.float32))
report = {"tag": args.tag, "anchors": len(TR), "anchor_stride": args.anchor_stride,
          "learning_rate": args.learning_rate, "rounds": rounds, "leaves": args.leaves,
          "seeds": args.seeds, "direct_weight": args.direct_weight,
          "fold378_calibrated_rmsle": score, "elapsed_s": time.time() - started}
json.dump(report, open(f"{ROOT}/work/{args.tag}_report.json", "w"), indent=2)
print(json.dumps(report, indent=2), flush=True)
print(f"FOLD378 {args.tag}: {score:.9f}", flush=True)
