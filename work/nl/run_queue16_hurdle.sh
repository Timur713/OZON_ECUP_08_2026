#!/bin/bash
# 225 - the hurdle as TWO SEPARATE NETWORKS on the confirmed representation.
# 82 percent of the achievable variance is the buy event (FINDINGS 5). The pool
# has hurdle heads inside one network, and it has cls classifiers on the
# ordinary representation which failed. It has never had two separate networks,
# each specialised on one factor, built on the shape representation that is the
# only representation confirmed to work on public.
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
run divV_prob_shape  --seed 1101 --variant decay --objective probability --per-user-scale --epochs 4
run divV_magn_shape  --seed 1102 --variant decay --objective magnitude --per-user-scale --epochs 4
run divV_prob_plain  --seed 1103 --variant decay --objective probability --epochs 4
run divV_magn_plain  --seed 1104 --variant decay --objective magnitude --epochs 4
run divV_prob_st24   --seed 1105 --variant decay --objective probability --anchor-stride 24 --epochs 4
run divV_magn_st24   --seed 1106 --variant decay --objective magnitude --anchor-stride 24 --epochs 4
echo "QUEUE16_DONE $(date -u +%H:%M:%S)"
