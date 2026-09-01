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

echo "WAIT existing survival selection $(date --iso-8601=seconds)"
while pgrep -f '[t]rain_classifier_gpu.py surv409_select342' >/dev/null; do
  sleep 30
done

old_supervisor=$(cat "$ROOT/work/server_survival_growth_old.pid")
kill -KILL "$old_supervisor" 2>/dev/null || true

selected_epoch=$(
  "$PY" - "$ROOT/work/surv409_select342_history.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
if len(rows) != 3:
    raise SystemExit(f"expected 3 selection epochs, received {len(rows)}")
print(min(rows, key=lambda row: row["score"])["epoch"])
PY
)
echo "SELECTED epoch=$selected_epoch from fold 342"

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed 1300
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --calendar --survival-head --private-selection --anchor-start 43
)
echo "START untouched survival fold 378 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" surv409_holdout378 "${COMMON[@]}" --val 378 --epochs 3 \
  > "$LOGS/surv409_holdout378.log" 2>&1

"$PY" "$BUILD" > "$LOGS/surv409_growth_build.log" 2>&1
cat "$LOGS/surv409_growth_build.log"
echo "SURVIVAL_GROWTH_DONE $(date --iso-8601=seconds)"
