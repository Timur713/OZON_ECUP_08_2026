#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT" ECUP_MAT="$ROOT/work/mat" ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
run () {
  tag=$1; shift
  [ -s "$ROOT/work/${tag}_val.npy" ] && { echo "SKIP $tag"; return; }
  echo "START $tag $(date --iso-8601=seconds)"
  "$PY" work/train_exact_scale.py "$tag" --variant plain --seed 93 --epochs 2 "$@" \
    > "$ROOT/logs/${tag}.log" 2>&1 || { echo "FAIL $tag"; return; }
  "$PY" work/evaluate_validation_ridge.py \
    "$ROOT/work/w409c_val.npy" "$ROOT/work/w409_exact_decay_s93_val.npy" \
    "$ROOT/work/${tag}_val.npy" --joint --repeats 96 \
    > "$ROOT/work/${tag}_admitted_joint96.json" 2>/dev/null
  echo "DONE $tag $(grep -h DONE "$ROOT/logs/${tag}.log" | tail -1)"
}
run rv_search    --window 409 --channel-list "searches,search,gmv_search,search_to_cart,search_to_ord,has_search_to_ord,has_search_to_cart"
run rv_cat       --window 409 --channel-list "cat,gmv_cat,cat_to_cart,cat_to_ord,has_cat_to_ord,has_cat_to_cart"
run rv_noamount  --window 409 --channel-list "to_ord,to_cart,searches,active,search,cat"
run rv_activity  --window 409 --channel-list "active,searches"
run rv_intent    --window 409 --channel-list "to_cart,to_ord,search_to_cart,search_to_ord,cat_to_cart,cat_to_ord"
run rv_gmvonly   --window 409 --channel-list "gmv"
run sw28         --window 28
run sw14         --window 14
echo "RESTRICTED_DONE $(date --iso-8601=seconds)"
