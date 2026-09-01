#!/bin/bash
# 196 - CONDITIONAL queue: launch only if divM_plain_e4 shows that the gain
# belongs to the SCHEDULE rather than to the shape representation.
#
# If four epochs helps the ordinary representation too, then every network in
# the pool is undertrained at its standard two epochs, and retraining the
# strongest recipes at a longer schedule is worth more than any new axis. That
# is the one scenario left in which the gap to top-3 closes by more than a
# thousandth rather than by a ten-thousandth at a time.
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
# The two admitted recipes retrained on the longer schedule, several seeds each,
# so the ridge sees genuinely new members of the families it already trusts.
run divN_decay_e4_s1 --seed 401 --variant decay --epochs 4
run divN_decay_e4_s2 --seed 402 --variant decay --epochs 4
run divN_pos_e4_s1   --seed 403 --variant position --epochs 4
run divN_pos_e4_s2   --seed 404 --variant position --epochs 4
run divN_decay_e6_s1 --seed 405 --variant decay --epochs 6
run divN_pos_e6_s1   --seed 406 --variant position --epochs 6
run divN_posdecay_e4 --seed 407 --variant position_decay --epochs 4
run divN_buyer_e4    --seed 408 --variant buyer --epochs 4
echo "QUEUE9_DONE $(date -u +%H:%M:%S)"
