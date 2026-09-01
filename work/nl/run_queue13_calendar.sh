#!/bin/bash
# 216 - give the networks a calendar.
# Every network in the pool reads "the last 409 days" with no absolute date. It
# therefore cannot know WHICH thirty days it is being asked to predict, and can
# only learn an averaged seasonal response across the anchors it saw, which span
# target windows from August 2025 to February 2026. The competition window is a
# gift peak, and the model has no way to represent that.
#
# The boosting bases already receive the anchor's day of year through feats4
# line 98. The networks, which carry most of the ensemble's weight, never have.
# This is a structural gap, not a hyperparameter.
#
# Four channels: sine and cosine of day-of-year and of day-of-week, per day.
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
run divS_cal_e2      --seed 801 --variant decay --calendar --epochs 2
run divS_cal_e4      --seed 802 --variant decay --calendar --epochs 4
run divS_cal_e2_s2   --seed 803 --variant decay --calendar --epochs 2
run divS_cal_pos_e2  --seed 804 --variant position --calendar --epochs 2
run divS_cal_shape_e4 --seed 805 --variant decay --calendar --per-user-scale --epochs 4
run divS_cal_e4_s2   --seed 806 --variant decay --calendar --epochs 4
# Widen the anchor span so the network sees more distinct seasonal positions,
# which is what makes a calendar input learnable at all.
run divS_cal_wide_e4 --seed 807 --variant decay --calendar --anchor-start 130 --valid-channel --epochs 4
run divS_cal_wide_e6 --seed 808 --variant decay --calendar --anchor-start 90 --valid-channel --epochs 6
echo "QUEUE13_DONE $(date -u +%H:%M:%S)"
