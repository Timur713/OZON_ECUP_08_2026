#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT" ECUP_MAT="$ROOT/work/mat" ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
for variant in plain decay; do
  for seed in 93 3141 5150 9090 2718; do
    tag="exext_${variant}_s${seed}"
    [ -s "$ROOT/work/${tag}_final.npy" ] && { echo "SKIP $tag"; continue; }
    echo "START $tag $(date --iso-8601=seconds)"
    "$PY" work/train_exact_scale.py "$tag" --variant "$variant" --window 409 --seed "$seed" \
      --epochs 2 --anchor-start 186 --anchor-end 378 \
      > "$ROOT/logs/${tag}.log" 2>&1 || { echo "FAIL $tag"; continue; }
    echo "DONE $tag $(grep -h DONE "$ROOT/logs/${tag}.log" | tail -1)"
  done
done
"$PY" - <<'PYEOF'
import json
import numpy as np
W="/home/ubuntu/ecup/work/"
for variant in ("plain","decay"):
    parts=[];used=[]
    for s in (93,3141,5150,9090,2718):
        try:
            parts.append(np.load(f"{W}exext_{variant}_s{s}_final.npy").astype(np.float64)); used.append(s)
        except FileNotFoundError: pass
    if parts:
        np.save(f"{W}EXEXT_{variant}_final.npy", np.mean(parts,0).astype(np.float32))
        json.dump({"variant":variant,"seeds":used,"anchors":"186..378"},
                  open(f"{W}EXEXT_{variant}_manifest.json","w"),indent=2)
        print(variant,"averaged",len(parts),"seeds",used)
PYEOF
echo "EXTENDED_DONE $(date --iso-8601=seconds)"
