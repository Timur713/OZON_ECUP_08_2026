#!/bin/bash
# 173 v2 - reordered after the first result. Extending the anchor range back to
# day 43 makes the model WORSE (calibrated 1.66985 against the matched control
# 1.66796) because a 409-day window at an early anchor is mostly left padding,
# which is an input distribution the competition anchor never presents. The
# clean axes run first; the extended range returns only with a validity channel
# that tells the network which days are real.
set -u
cd /home/ubuntu/ecup3
export ECUP_ROOT=/home/ubuntu/ecup3
export ECUP_OUT=/home/ubuntu/ecup3/work/div
PY=./.venv/bin/python
mkdir -p work/div
run () {
  local tag="$1"; shift
  if [ -f "work/div/${tag}_final.npy" ]; then echo "SKIP $tag"; return; fi
  echo "=== START $tag $* $(date -u +%H:%M:%S) ==="
  $PY work/train_w409_diverse.py "$tag" "$@" 2>&1 | tail -4
  echo "=== END $tag $(date -u +%H:%M:%S) ==="
}

# B: disjoint anchor phases inside the untruncated range. No padding shift.
run divB_st24a      --seed 211 --variant decay --anchor-stride 24
run divB_st24b      --seed 212 --variant decay --anchor-stride 24 --anchor-phase 12
run divB_st36a      --seed 213 --variant decay --anchor-stride 36
run divB_st36b      --seed 214 --variant decay --anchor-stride 36 --anchor-phase 12
run divB_st36c      --seed 215 --variant decay --anchor-stride 36 --anchor-phase 24

# C: bagging over users.
run divC_u50a       --seed 221 --variant decay --user-fraction 0.5 --user-seed 1
run divC_u50b       --seed 222 --variant decay --user-fraction 0.5 --user-seed 2
run divC_u33a       --seed 223 --variant decay --user-fraction 0.33 --user-seed 3

# E: the extended range done properly, with a channel marking real days.
run divE_s43_valid  --seed 241 --variant decay --anchor-start 43  --anchor-stride 12 --valid-channel
run divE_s90_valid  --seed 242 --variant decay --anchor-start 90  --anchor-stride 12 --valid-channel
run divE_s130_valid --seed 243 --variant decay --anchor-start 130 --anchor-stride 12 --valid-channel
run divE_base_valid --seed 244 --variant decay --anchor-start 186 --anchor-stride 12 --valid-channel

# F: crosses of whichever clean axes exist, plus a longer schedule on the
# extended range, which only makes sense once padding is marked.
run divF_s43_valid_e3 --seed 251 --variant decay --anchor-start 43 --anchor-stride 12 --valid-channel --epochs 3
run divF_s43_valid_u50 --seed 252 --variant decay --anchor-start 43 --anchor-stride 12 --valid-channel --user-fraction 0.5 --user-seed 6
run divF_st24a_u50  --seed 253 --variant decay --anchor-stride 24 --user-fraction 0.5 --user-seed 7
run divF_s43_valid_pos --seed 254 --variant position --anchor-start 43 --anchor-stride 12 --valid-channel
echo "QUEUE_DONE $(date -u +%H:%M:%S)"
