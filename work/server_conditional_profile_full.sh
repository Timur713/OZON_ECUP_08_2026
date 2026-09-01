#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
AUDIT="$ROOT/work/evaluate_validation_ridge.py"
DECIDE="$ROOT/work/evaluate_profile_promotion.py"
AVERAGE="$ROOT/work/average_profile_finals.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

finalize_pid=$(cat "$ROOT/work/server_five_hour_finalize.pid")
echo "WAIT frozen profile audits pid=$finalize_pid $(date --iso-8601=seconds)"
while kill -0 "$finalize_pid" 2>/dev/null; do
  sleep 30
done
if [ ! -s "$ROOT/work/control409s2718_growth_report.json" ]; then
  echo "autonomous profile audit ended without reports" >&2
  exit 1
fi

# Recompute every promotion input on the server in one environment.  These are
# validation-label audits only and do not read any leaderboard score.
"$PY" "$AUDIT" "$ROOT/work/w409c_val.npy" --repeats 96 \
  > "$ROOT/work/w409c_ridge96_current.json"
for tag in control409_growth reg409_growth mark409_growth; do
  "$PY" "$AUDIT" "$ROOT/work/${tag}_val.npy" --repeats 96 \
    > "$ROOT/work/${tag}_ridge96.json"
  "$PY" "$AUDIT" "$ROOT/work/w409c_val.npy" "$ROOT/work/${tag}_val.npy" \
    --joint --repeats 96 > "$ROOT/work/${tag}_w409c_joint96.json"
done

"$PY" "$DECIDE" > "$LOGS/profile_promotion_decision.log" 2>&1
cat "$LOGS/profile_promotion_decision.log"

trainer_for() {
  case "$1" in
    control) echo "$ROOT/work/train_classifier_gpu_event_frozen.py" ;;
    regularity) echo "$ROOT/work/train_classifier_gpu_regularity_frozen.py" ;;
    marked) echo "$ROOT/work/train_classifier_gpu_marked_frozen.py" ;;
    *) return 1 ;;
  esac
}
validation_stem() {
  local family=$1 seed=$2
  case "$family:$seed" in
    control:1310) echo control409_select342 ;;
    control:2718) echo control409s2718_select342 ;;
    regularity:1310) echo reg409_select342 ;;
    regularity:2718) echo reg409s2718_select342 ;;
    marked:1310) echo mark409_select342 ;;
    marked:2718) echo mark409s2718_select342 ;;
    *) return 1 ;;
  esac
}
mapfile -t eligible < <("$PY" - "$ROOT/work/profile_promotion_decision.json" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["full_refit_families"]:
    print(value)
PY
)

for family in "${eligible[@]}"; do
  trainer=$(trainer_for "$family")
  finals=()
  for seed in 1310 2718; do
    selection=$(validation_stem "$family" "$seed")
    read -r epoch mix < <("$PY" - "$ROOT/work/${selection}_history.json" <<'PY'
import json, sys
row=min(json.load(open(sys.argv[1])), key=lambda value: value["score"])
print(row["epoch"], row["best_hurdle_weight"])
PY
)
    tag="promote_${family}_${seed}_full"
    extra=()
    if [ "$family" != control ]; then extra+=(--event-profile); fi
    echo "START promoted full family=$family seed=$seed epoch=$epoch mix=$mix $(date --iso-8601=seconds)"
    "$PY" "$trainer" "$tag" --mode final --window 409 --width 256 --blocks 8 \
      --seed "$seed" --stride 4 --frac 0.25 --channels all --bs 2048 \
      --pred-bs 2048 --summary --calendar --anchor-start 43 \
      --epochs "$epoch" --mix "$mix" "${extra[@]}" \
      > "$LOGS/${tag}.log" 2>&1
    finals+=("$ROOT/work/${tag}_final.npy")
  done

  "$PY" "$AVERAGE" "$family" "${finals[0]}" "${finals[1]}" \
    > "$LOGS/promote_${family}_average.log" 2>&1
  echo "READY seed-average vector family=$family $(date --iso-8601=seconds)"
done
echo "CONDITIONAL_PROFILE_FULL_DONE refit_families=${#eligible[@]} $(date --iso-8601=seconds)"
