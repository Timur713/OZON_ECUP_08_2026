#!/bin/bash
# 184 - a shape-only representation.
# Level is the dominant signal in this problem and every base in the pool is
# built around it. Dividing each user's window by that user's own mean removes
# the level entirely and forces the network onto the SHAPE of the history. The
# ridge still has all the level-carrying bases, so a shape model does not have
# to beat them; it only has to be different from them.
set -u
cd /home/ubuntu/ecup3
export ECUP_ROOT=/home/ubuntu/ecup3
export ECUP_OUT=/home/ubuntu/ecup3/work/div
PY=./.venv/bin/python
run () {
  local tag="$1"; shift
  if [ -f "work/div/${tag}_final.npy" ]; then echo "SKIP $tag"; return; fi
  echo "=== START $tag $* $(date -u +%H:%M:%S) ==="
  $PY work/train_w409_diverse.py "$tag" "$@" 2>&1 | tail -5
  echo "=== END $tag $(date -u +%H:%M:%S) ==="
}
run divK_shape      --seed 301 --variant decay --per-user-scale
run divK_shape_e4   --seed 302 --variant decay --per-user-scale --epochs 4
run divK_shape_st24 --seed 303 --variant decay --per-user-scale --anchor-stride 24
run divK_shape_pos  --seed 304 --variant position --per-user-scale
echo "QUEUE6_DONE $(date -u +%H:%M:%S)"
