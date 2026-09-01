#!/usr/bin/env bash
set -euo pipefail

ROOT=${ECUP_ROOT:-/Users/timur/Desktop/dev/OZON_ECUP_2026_3}
tag=${1:?usage: freeze_profile_probe_gate.sh TAG}
PY="$ROOT/.venv/bin/python"

# The frozen measured/admitted block. All scores predate every
# 124+ probe.  The 0.00004 adaptive charge is fixed per structural candidate;
# it is added on top of twice the empirical df transfer cost.
"$PY" "$ROOT/work/calculate_probe_gates.py" "$tag" \
  --extra 83_probe_cls300 1.6488394251718939 \
  --extra 86_probe_cls300_probability 1.6558577069 \
  --extra 85_probe_w210a 1.6482434279349687 \
  --extra 89_probe_w300a 1.6472946857056134 \
  --extra 92_probe_cls409_r26 1.647041762499095 \
  --extra 102_probe_w409c 1.646720938726788 \
  --lam 0.003 --adaptive-cost 0.00004 \
  > "$ROOT/work/${tag}_gates.json"
"$PY" - "$ROOT/work/${tag}_gates.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
if value["strict_with_adaptive_gate"] is None:
    raise SystemExit("no positive-weight strict+adaptive gate")
print(json.dumps({
    "probe": value["probe_tag"],
    "strict_with_adaptive_gate": value["strict_with_adaptive_gate"],
    "positive_gate": value["positive_empirical_net_gate"],
    "neutral_score": value["positive_weight_neutral_score"],
}, indent=2))
PY
