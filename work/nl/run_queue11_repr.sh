#!/bin/bash
# 207 - new REPRESENTATIONS, not new seeds.
# The shape representation is confirmed on public: +0.00022 realised. But
# shape_e8 shows that another draw of the SAME recipe overlaps almost entirely
# with the base already admitted, dropping from +0.0003124 against the old pool
# to +0.0000958 against the pool that contains shape_e4. So seeds are not the
# lever; genuinely different views of the same history are.
#
# Every run uses four epochs, because the one thing the round established is
# that a changed representation needs a longer schedule than the pool's standard
# two before its orthogonal component carries signal.
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
# daynorm divides each day by the population mean for that day, so the network
# sees behaviour with calendar seasonality removed. Seasonality of the target
# window is the hard part of this problem, which makes this the most promising
# of the five.
run divQ_daynorm_e4    --seed 601 --variant decay --representation daynorm --epochs 4
run divQ_cumulative_e4 --seed 602 --variant decay --representation cumulative --epochs 4
run divQ_diff_e4       --seed 603 --variant decay --representation diff --epochs 4
run divQ_occurrence_e4 --seed 604 --variant decay --representation occurrence --epochs 4
# Cross the two representations that remove different things: level and calendar.
run divQ_daynorm_shape_e4 --seed 605 --variant decay --representation daynorm --per-user-scale --epochs 4
run divQ_daynorm_e6    --seed 606 --variant decay --representation daynorm --epochs 6
run divQ_cumulative_e6 --seed 607 --variant decay --representation cumulative --epochs 6
run divQ_rankday_e4    --seed 608 --variant decay --representation rankday --epochs 4
echo "QUEUE11_DONE $(date -u +%H:%M:%S)"
