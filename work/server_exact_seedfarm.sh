#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT" ECUP_MAT="$ROOT/work/mat" ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

# wait for the 17-channel test to release the GPU
while pgrep -f "train_exact_scale.py exch17" > /dev/null; do sleep 20; done

for seed in 2718 3141 4242 5150 6060 7070 8080 9090; do
  tag="exp409_s${seed}"
  [ -s "$ROOT/work/${tag}_val.npy" ] && { echo "SKIP $tag"; continue; }
  echo "START $tag $(date --iso-8601=seconds)"
  "$PY" work/train_exact_scale.py "$tag" --variant plain --window 409 \
    --seed "$seed" --epochs 2 > "$ROOT/logs/${tag}.log" 2>&1 || { echo "FAIL $tag"; continue; }
  echo "DONE $tag $(grep -h DONE "$ROOT/logs/${tag}.log" | tail -1)"
done

echo "BUILD seed average $(date --iso-8601=seconds)"
"$PY" - <<'PYEOF'
import glob, json
import numpy as np
ROOT="/home/ubuntu/ecup/work/"
tags=["exw409_s93"]+[f"exp409_s{s}" for s in (2718,3141,4242,5150,6060,7070,8080,9090)]
for kind in ("val","final"):
    parts=[]
    used=[]
    for t in tags:
        p=f"{ROOT}{t}_{kind}.npy"
        try:
            parts.append(np.load(p).astype(np.float64)); used.append(t)
        except FileNotFoundError:
            pass
    if parts:
        np.save(f"{ROOT}EXP409AVG_{kind}.npy", np.mean(parts,0).astype(np.float32))
        print(kind,"averaged",len(parts),"seeds:",used)
        json.dump({"kind":kind,"seeds":used,"count":len(parts)},
                  open(f"{ROOT}EXP409AVG_{kind}_manifest.json","w"),indent=2)
PYEOF

"$PY" work/evaluate_validation_ridge.py "$ROOT/work/EXP409AVG_val.npy" \
  --repeats 96 > "$ROOT/work/EXP409AVG_ridge96.json" 2>/dev/null
"$PY" work/evaluate_validation_ridge.py \
  "$ROOT/work/w409c_val.npy" "$ROOT/work/w409_exact_decay_s93_val.npy" \
  "$ROOT/work/EXP409AVG_val.npy" --joint --repeats 96 \
  > "$ROOT/work/EXP409AVG_admitted_joint96.json" 2>/dev/null
echo "SEEDFARM_DONE $(date --iso-8601=seconds)"
