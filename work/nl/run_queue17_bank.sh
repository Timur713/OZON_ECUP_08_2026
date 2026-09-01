#!/bin/bash
# 232 - grow the model bank.
# Fourteen axes have been tested and two passed. But adding ALL 51 models to the
# pool moved the fold-378 out-of-fold score by 0.00056, so the value is in
# collective span rather than in any single base, and the composite converts
# quantity into one admissible base at the cost of one probe.
#
# Individual admissibility is therefore no longer the objective for these runs.
# The objective is coverage: spread seeds and configurations widely, including
# across axes that failed individually, because a model that is useless alone
# can still widen the span.
set -u
cd /home/ubuntu/ecup3
export ECUP_ROOT=/home/ubuntu/ecup3
export ECUP_OUT=/home/ubuntu/ecup3/work/div
PY=./.venv/bin/python
run () {
  local tag="$1"; shift
  if [ -f "work/div/${tag}_final.npy" ]; then echo "SKIP $tag"; return; fi
  echo "=== START $tag $* $(date -u +%H:%M:%S) ==="
  $PY work/train_w409_diverse.py "$tag" "$@" 2>&1 | tail -3
  echo "=== END $tag $(date -u +%H:%M:%S) ==="
}
for s in 2001 2002 2003; do
  run "bank_shape_e4_s${s}"  --seed "$s" --variant decay --per-user-scale --epochs 4
done
for s in 2011 2012 2013; do
  run "bank_st24_s${s}"      --seed "$s" --variant decay --anchor-stride 24
done
for s in 2021 2022; do
  run "bank_st36_s${s}"      --seed "$s" --variant decay --anchor-stride 36
done
for s in 2031 2032; do
  run "bank_daily_s${s}"     --seed "$s" --variant decay --daily-head 0.3 --epochs 4
done
for s in 2041 2042; do
  run "bank_pos_s${s}"       --seed "$s" --variant position --epochs 2
done
for s in 2051 2052; do
  run "bank_decay_s${s}"     --seed "$s" --variant decay --epochs 2
done
run bank_cal_s2061   --seed 2061 --variant decay --calendar --epochs 2
run bank_diff_s2071  --seed 2071 --variant decay --representation diff --epochs 4
run bank_cum_s2081   --seed 2081 --variant decay --representation cumulative --epochs 4
run bank_occ_s2091   --seed 2091 --variant decay --representation occurrence --epochs 4
echo "QUEUE17_DONE $(date -u +%H:%M:%S)"
