#!/usr/bin/env bash
# Deliberately overlong MPS tail. Strong 409 replications come first; every
# completed epoch writes validation/final vectors, so the queue is safe to stop.
set -u

ROOT=/Users/timur/Desktop/dev/OZON_ECUP_2026_3
PY=$ROOT/.venv/bin/python
LOG=$ROOT/work/offline_tail.log
cd "$ROOT"

guard() {
  local free_gb
  free_gb=$(df -m /Users/timur | tail -1 | awk '{print int($4/1024)}')
  if [[ $free_gb -lt 6 ]]; then
    echo "ABORT: only ${free_gb}GB disk left"
    exit 1
  fi
}

run_model() {
  local tag=$1 arch=$2 window=$3 epochs=$4 seed=$5 head=$6
  guard
  date
  echo "START $tag arch=$arch window=$window epochs=$epochs seed=$seed head=$head"
  if "$PY" "$ROOT/work/train_seq2.py" \
      "$tag" "$arch" "$window" "$epochs" "$seed" "$head" >>"$LOG" 2>&1; then
    echo "DONE $tag"
  else
    echo "FAILED $tag"
  fi
}

echo "Waiting for queue_offline_followup"
while pgrep -f '[q]ueue_offline_followup.sh' >/dev/null; do sleep 30; done

# w409c is the only local family with a large, stable 50k->200k ridge gain.
run_model w409d tcn 409 2 119 direct
run_model w409e tcn 409 2 120 direct

# Replications across useful information horizons; these are ensemble evidence,
# not automatic public probes.
run_model w300b tcn 300 2 121 direct
run_model w210c tcn 210 2 122 direct
run_model w365c tcn 365 2 123 direct

# Final over-capacity tail: an independently testable hurdle head at 409 days.
run_model w409twoa tcn 409 2 124 two

date
echo OFFLINE_TAIL_DONE
