#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
REMOTE=/home/ubuntu/ecup
SSH=(ssh -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "$HOST")

cd "$ROOT"
while ! "${SSH[@]}" "test -s $REMOTE/work/server_pairsafe_residual.pid"; do
  sleep 15
done
while "${SSH[@]}" "kill -0 \$(cat $REMOTE/work/server_pairsafe_residual.pid) 2>/dev/null"; do
  "${SSH[@]}" "date; tail -1 $REMOTE/logs/pairsafe_residual_supervisor.log 2>/dev/null || true; tail -2 $REMOTE/logs/residcurve_pairsafe.log 2>/dev/null || true"
  sleep 30
done

if ! "${SSH[@]}" "test -s $REMOTE/work/residcurve_pairsafe_report.json"; then
  echo "pair-safe residual pipeline stopped without a report" >&2
  "${SSH[@]}" "tail -100 $REMOTE/logs/pairsafe_residual_supervisor.log 2>/dev/null || true; tail -100 $REMOTE/logs/residcurve_pairsafe.log 2>/dev/null || true"
  exit 1
fi

scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
  "$HOST:$REMOTE/work/residcurve_pairsafe_report.json" \
  "$HOST:$REMOTE/work/residcurve_pairsafe_residual_pairs.npy" \
  "$ROOT/work/"

echo "pair-safe residual artifacts fetched"
