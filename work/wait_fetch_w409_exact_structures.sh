#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
REMOTE=/home/ubuntu/ecup
SSH=(ssh -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "$HOST")

while ! "${SSH[@]}" "test -s $REMOTE/work/server_w409_exact_structures.pid"; do sleep 10; done
while "${SSH[@]}" "kill -0 \$(cat $REMOTE/work/server_w409_exact_structures.pid) 2>/dev/null"; do sleep 30; done
scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
  "$HOST:$REMOTE/work/w409_exact_structures_decision.json" "$ROOT/work/"
for variant in position event; do
  passed=$("$ROOT/.venv/bin/python" - "$ROOT/work/w409_exact_structures_decision.json" "$variant" <<'PY'
import json, sys
rows=json.load(open(sys.argv[1]))["variants"]
print("1" if next(row for row in rows if row["variant"] == sys.argv[2])["passed"] else "0")
PY
)
  if [ "$passed" = 1 ]; then
    tag="w409_exact_${variant}_s93"
    scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
      "$HOST:$REMOTE/work/${tag}_final.npy" "$ROOT/work/"
    number=131; if [ "$variant" = event ]; then number=132; fi
    probe="${number}_probe_w409_exact_${variant}_s93"
    ECUP_ROOT="$ROOT" "$ROOT/.venv/bin/python" "$ROOT/work/build_gpu_probe.py" \
      "$probe" "$ROOT/work/${tag}_final.npy" \
      --base-submission submissions/130_private_safe_exact_decay_l003.csv \
      --base-score 1.6461706600883055 --weight 0.30
    ECUP_ROOT="$ROOT" bash "$ROOT/work/freeze_profile_probe_gate.sh" "$probe"
    "$ROOT/.venv/bin/python" "$ROOT/work/validate_submissions.py" \
      "$ROOT/submissions/${probe}.csv"
  fi
done
echo "exact structure decisions fetched"
