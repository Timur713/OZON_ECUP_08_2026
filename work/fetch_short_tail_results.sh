#!/usr/bin/env bash
# Fetch short-sidecar results as soon as each full prediction is committed.
set -u

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 "$HOST")
SCP=(scp -q -i "$KEY" -o BatchMode=yes)

wait_and_fetch() {
  local tag=$1
  local remote_tag=${tag}_full
  while ! "${SSH[@]}" \
      "test -s /home/ubuntu/ecup/work/${remote_tag}_final.npy" 2>/dev/null; do
    if "${SSH[@]}" \
        "grep -q '^FAILED ${remote_tag}$' /home/ubuntu/ecup/logs/short_tail.log" \
        2>/dev/null; then
      date '+%Y-%m-%d %H:%M:%S %Z'
      echo "SKIP_FAILED ${remote_tag}"
      return 0
    fi
    sleep 30
  done
  "${SCP[@]}" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_final.npy" \
    "$ROOT/work/${remote_tag}_server_final.npy"
  "${SCP[@]}" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_final_components.npz" \
    "$ROOT/work/${remote_tag}_server_components.npz"
  for suffix in config.json history.json; do
    "${SCP[@]}" \
      "$HOST:/home/ubuntu/ecup/work/${remote_tag}_${suffix}" \
      "$ROOT/work/${remote_tag}_${suffix}"
  done
  for remote_suffix in val.npy best_val_components.npz; do
    if "${SSH[@]}" \
        "test -s /home/ubuntu/ecup/work/${tag}_val_${remote_suffix}"; then
      local_suffix=$remote_suffix
      [[ $remote_suffix == val.npy ]] && local_suffix=server_val.npy
      [[ $remote_suffix == best_val_components.npz ]] \
        && local_suffix=server_best_val_components.npz
      "${SCP[@]}" \
        "$HOST:/home/ubuntu/ecup/work/${tag}_val_${remote_suffix}" \
        "$ROOT/work/${tag}_val_${local_suffix}"
    fi
  done
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "FETCHED ${remote_tag}"
}

for tag in cls120hyb cls120hybmkt cls180hybmkt cls90hybmkt cls60hybmkt cls240hyb; do
  wait_and_fetch "$tag"
done

echo SHORT_TAIL_FETCH_DONE
