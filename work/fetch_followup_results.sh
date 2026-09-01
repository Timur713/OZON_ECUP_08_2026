#!/usr/bin/env bash
set -u

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 "$HOST")
SCP=(scp -i "$KEY" -o BatchMode=yes)

fetch_candidate() {
  local tag=$1
  local supervisor_log=${2:-followup_supervisor.log}
  local remote_tag=${tag}_full
  local status
  while true; do
    status=$("${SSH[@]}" \
      "grep -Eh '^(DONE|FAILED|SKIP|SKIP_AFTER_FAILURE) ${remote_tag}$' /home/ubuntu/ecup/logs/${supervisor_log} | tail -1" 2>/dev/null || true)
    [[ -n $status ]] && break
    sleep 30
  done
  if [[ $status != "DONE ${remote_tag}" ]]; then
    echo "$status (not fetched)"
    return 0
  fi
  "${SCP[@]}" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_final.npy" \
    "$ROOT/work/${remote_tag}_server_final.npy"
  "${SCP[@]}" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_final_components_hurdle.npy" \
    "$ROOT/work/${remote_tag}_server_hurdle.npy"
  "${SCP[@]}" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_final_components.npz" \
    "$ROOT/work/${remote_tag}_server_components.npz"
  "${SCP[@]}" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_config.json" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_history.json" \
    "$ROOT/work/"
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/${tag}_val_best_val_components.npz"; then
    "${SCP[@]}" \
      "$HOST:/home/ubuntu/ecup/work/${tag}_val_best_val_components.npz" \
      "$ROOT/work/${tag}_val_server_best_val_components.npz"
  fi
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/${tag}_val_val.npy"; then
    "${SCP[@]}" \
      "$HOST:/home/ubuntu/ecup/work/${tag}_val_val.npy" \
      "$ROOT/work/${tag}_val_server_val.npy"
  fi
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "FETCHED $remote_tag"
}

fetch_candidate cls300mkt extra_supervisor.log

for tag in cls43gift cls409hyb cls409dense cls409mkt cls120hyb cls300hybmkt cls30 cls240; do
  fetch_candidate "$tag"
done

echo "FOLLOWUP_FETCH_DONE"
