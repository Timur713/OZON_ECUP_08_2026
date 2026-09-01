#!/bin/bash
# 217 - the combination that makes the 2025 gift season learnable.
#
# Three facts that only matter together:
#  1. The last training target window ends on day 378, 13 February 2026. The
#     competition window is days 409-438, 14 February to 15 March. Every anchor
#     the pool trains on therefore predicts a season the competition window is
#     NOT in, so a calendar input alone would have to extrapolate.
#  2. The data does contain that season: days 44-73 are 14 February to 15 March
#     2025, reachable from anchor 43.
#  3. Anchor 43 failed twice before. Without a validity channel its 409-day
#     window is 90 percent left padding. With one, the model was fine but the
#     anchor still added nothing, because the network had no way to represent
#     WHICH season it was looking at and so could not tell that anchor apart.
#
# Calendar plus validity plus the extended range is the first configuration in
# which the same-season anchor is representable at all.
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
run divT_gift_e4  --seed 901 --variant decay --calendar --valid-channel --anchor-start 43 --epochs 4
run divT_gift_e6  --seed 902 --variant decay --calendar --valid-channel --anchor-start 43 --epochs 6
run divT_gift_e4b --seed 903 --variant decay --calendar --valid-channel --anchor-start 43 --epochs 4
run divT_gift_pos --seed 904 --variant position --calendar --valid-channel --anchor-start 43 --epochs 4
run divT_gift_shape --seed 905 --variant decay --calendar --valid-channel --anchor-start 43 --per-user-scale --epochs 4
run divT_gift_st6 --seed 906 --variant decay --calendar --valid-channel --anchor-start 43 --anchor-stride 6 --epochs 4
echo "QUEUE14_DONE $(date -u +%H:%M:%S)"
