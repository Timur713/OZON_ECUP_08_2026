#!/bin/bash
# 201 - scale the one axis that is now CONFIRMED ON PUBLIC.
# 191_probe_shape_e4 passed its frozen gate by 0.00019 and realised +0.00022 of
# public. The calibrated score of that recipe varies a lot by seed (1.667826,
# 1.672188, 1.671043), so seeds are not clones here the way they are for the
# ordinary representation: each draw is its own candidate base. The cross with
# the sparse-anchor axis, which also passed on public, is included because those
# two are nearly additive offline.
set -u
cd /home/ubuntu/ecup3
export ECUP_ROOT=/home/ubuntu/ecup3
export ECUP_OUT=/home/ubuntu/ecup3/work/div
PY=./.venv/bin/python
run () {
  local tag="$1"; shift
  if [ -f "work/div/${tag}_final.npy" ]; then echo "SKIP $tag"; return; fi
  echo "=== START $tag $* $(date -u +%H:%M:%S) ==="
  $PY work/train_w409_diverse.py "$tag" "$@" 2>&1 | tail -4
  echo "=== END $tag $(date -u +%H:%M:%S) ==="
}
for s in 501 502 503 504 505 506; do
  run "divP_shape_e4_s${s}" --seed "$s" --variant decay --per-user-scale --epochs 4
done
for s in 511 512 513; do
  run "divP_shape_e4_st24_s${s}" --seed "$s" --variant decay --per-user-scale --epochs 4 --anchor-stride 24
done
for s in 521 522; do
  run "divP_shape_e4_pos_s${s}" --seed "$s" --variant position --per-user-scale --epochs 4
done
for s in 531 532; do
  run "divP_shape_e4_st36_s${s}" --seed "$s" --variant decay --per-user-scale --epochs 4 --anchor-stride 36
done
echo "QUEUE10_DONE $(date -u +%H:%M:%S)"
