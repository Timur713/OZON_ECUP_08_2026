#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_classifier_gpu.py"
BUILD="$ROOT/work/build_survival_holdout.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

while pgrep -f '[s]erver_morning_priority.sh' >/dev/null; do
  echo "WAIT morning GPU queue $(date --iso-8601=seconds)"
  sleep 30
done

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed 1300
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --calendar --survival-head --private-selection --anchor-start 43
)

echo "START survival selection fold 342 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" surv409_select342 "${COMMON[@]}" --val 342 --epochs 3 \
  > "$LOGS/surv409_select342.log" 2>&1

selected_epoch=$(
  "$PY" - "$ROOT/work/surv409_select342_history.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
print(min(rows, key=lambda row: row["score"])["epoch"])
PY
)
echo "SELECTED epoch=$selected_epoch from fold 342"

echo "START untouched survival fold 378 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" surv409_holdout378 "${COMMON[@]}" --val 378 \
  --epochs 3 > "$LOGS/surv409_holdout378.log" 2>&1

"$PY" "$BUILD" > "$LOGS/surv409_growth_build.log" 2>&1
cat "$LOGS/surv409_growth_build.log"
echo "SURVIVAL_GROWTH_DONE $(date --iso-8601=seconds)"
