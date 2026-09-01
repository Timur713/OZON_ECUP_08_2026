#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
REMOTE=/home/ubuntu/ecup
SSH=(ssh -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "$HOST")
while ! "${SSH[@]}" "test -s $REMOTE/work/server_w409_exact_buyer.pid"; do sleep 10; done
while "${SSH[@]}" "kill -0 \$(cat $REMOTE/work/server_w409_exact_buyer.pid) 2>/dev/null"; do sleep 30; done
scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
  "$HOST:$REMOTE/work/w409_exact_buyer_decision.json" "$ROOT/work/"
passed=$("$ROOT/.venv/bin/python" - "$ROOT/work/w409_exact_buyer_decision.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1]))["passed"] else "0")
PY
)
if [ "$passed" = 1 ]; then
  scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
    "$HOST:$REMOTE/work/w409_exact_buyer_s93_final.npy" "$ROOT/work/"
  probe=133_probe_w409_exact_buyer_s93
  ECUP_ROOT="$ROOT" "$ROOT/.venv/bin/python" "$ROOT/work/build_gpu_probe.py" \
    "$probe" "$ROOT/work/w409_exact_buyer_s93_final.npy" \
    --base-submission submissions/130_private_safe_exact_decay_l003.csv \
    --base-score 1.6461706600883055 --weight 0.30
  ECUP_ROOT="$ROOT" bash "$ROOT/work/freeze_profile_probe_gate.sh" "$probe"
  "$ROOT/.venv/bin/python" "$ROOT/work/validate_submissions.py" \
    "$ROOT/submissions/${probe}.csv"
fi
echo "exact buyer decision fetched; passed=$passed"
