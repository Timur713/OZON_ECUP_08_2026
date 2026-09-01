#!/bin/bash
# 180 - push the anchor-subset axis to its limit.
# divB_st24a passed screening at +0.0001236 while being 0.0058 WORSE standalone
# than the matched control. That is FINDINGS 2.4 in action: a weak dissimilar
# model beats a strong similar one. The natural continuation is to make the
# anchor subset sparser still, which makes each model weaker and less like the
# full-anchor pool at the same time.
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
run divH_st48a --seed 271 --variant decay --anchor-stride 48
run divH_st48b --seed 272 --variant decay --anchor-stride 48 --anchor-phase 12
run divH_st48c --seed 273 --variant decay --anchor-stride 48 --anchor-phase 24
run divH_st48d --seed 274 --variant decay --anchor-stride 48 --anchor-phase 36
run divH_st72a --seed 275 --variant decay --anchor-stride 72
run divH_st72b --seed 276 --variant decay --anchor-stride 72 --anchor-phase 24
run divH_st72c --seed 277 --variant decay --anchor-stride 72 --anchor-phase 48
# Sparse anchors plus a user bag: two independent restrictions at once.
run divH_st48a_u50 --seed 278 --variant decay --anchor-stride 48 --user-fraction 0.5 --user-seed 11
run divH_st72a_u50 --seed 279 --variant decay --anchor-stride 72 --user-fraction 0.5 --user-seed 12
echo "QUEUE4_DONE $(date -u +%H:%M:%S)"
