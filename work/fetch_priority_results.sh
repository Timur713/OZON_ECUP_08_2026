#!/usr/bin/env bash
set -u

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 "$HOST")
SCP=(scp -i "$KEY" -o BatchMode=yes)
SUPERVISOR=/home/ubuntu/ecup/logs/morning_priority.log

fetch_candidate() {
  local tag=$1
  local remote_tag=${tag}_full
  local status
  while true; do
    status=$("${SSH[@]}" \
      "grep -Eh '^(DONE|FAILED|SKIP|SKIP_AFTER_FAILURE|PRESENT) ${remote_tag}$' '$SUPERVISOR' | tail -1" \
      2>/dev/null || true)
    [[ -n $status ]] && break
    sleep 30
  done
  if [[ $status == "PRESENT ${remote_tag}" ]]; then
    status="DONE ${remote_tag}"
  fi
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
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/${tag}_val_val_users.npy"; then
    "${SCP[@]}" \
      "$HOST:/home/ubuntu/ecup/work/${tag}_val_val_users.npy" \
      "$ROOT/work/${tag}_val_server_val_users.npy"
  fi
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "FETCHED ${remote_tag}"
}

for tag in cls300tprof cls43hold cls43holdtprof cls43exacthold cls43exactclass cls300tprofhead cls409hyb cls409dense cls409class cls43gift; do
  fetch_candidate "$tag"
done

while true; do
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/cls409wide_full_final.npy"; then
    fetch_candidate cls409wide
    break
  fi
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/cls409wide768_full_final.npy"; then
    fetch_candidate cls409wide768
    break
  fi
  if "${SSH[@]}" "grep -q '^SKIP cls409wide' '$SUPERVISOR'"; then
    echo "wide branch skipped"
    break
  fi
  sleep 30
done

for tag in cls120hyb cls409mkt cls120 cls300b cls60 cls300hybmkt cls30 cls240; do
  fetch_candidate "$tag"
done

echo "PRIORITY_FETCH_DONE"
