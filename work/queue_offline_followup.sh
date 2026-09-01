#!/usr/bin/env bash
set -u

ROOT=/Users/timur/Desktop/dev/OZON_ECUP_2026_3
PY=$ROOT/.venv/bin/python
LOG=$ROOT/work/offline_followup.log
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
  local tag=$1
  local window=$2
  local seed=$3
  guard
  date
  echo "START $tag window=$window seed=$seed"
  if "$PY" "$ROOT/work/train_seq2.py" "$tag" tcn "$window" 2 "$seed" direct >>"$LOG" 2>&1; then
    echo "DONE $tag"
  else
    echo "FAILED $tag"
  fi
}

echo "Waiting for the primary local queue"
while pgrep -f '[q]ueue_offline.sh' >/dev/null; do sleep 30; done

# Fill information-horizon gaps first, then add seeds to underrepresented
# windows. The queue is deliberately longer than the available night.
run_model w240a 240 111
run_model w336a 336 112
run_model w75a 75 113
run_model w30a 30 114
run_model w150b 150 115
run_model w210b 210 116
run_model w240b 240 117
run_model w336b 336 118

date
echo "OFFLINE_FOLLOWUP_DONE"
