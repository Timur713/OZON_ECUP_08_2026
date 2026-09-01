#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 "$HOST")
SCP=(scp -i "$KEY" -o BatchMode=yes)
PY=$ROOT/.venv/bin/python

fetch_candidate() {
  local remote_tag=$1
  local local_file=$ROOT/work/${remote_tag}_server_final.npy
  local component_file=$ROOT/work/${remote_tag}_server_hurdle.npy
  local component_archive=$ROOT/work/${remote_tag}_server_components.npz
  local val_tag=${remote_tag%_full}_val
  local remote_file=/home/ubuntu/ecup/work/${remote_tag}_final.npy
  local remote_components=/home/ubuntu/ecup/work/${remote_tag}_final_components_hurdle.npy
  while ! "${SSH[@]}" \
    "grep -qh '^DONE ${remote_tag}$' /home/ubuntu/ecup/logs/night_supervisor.log /home/ubuntu/ecup/logs/extra_supervisor.log"; do
    sleep 30
  done
  while ! "${SSH[@]}" "test -s '$remote_file'"; do sleep 30; done
  while ! "${SSH[@]}" "test -s '$remote_components'"; do sleep 30; done
  sleep 5
  "${SCP[@]}" "$HOST:$remote_file" "$local_file"
  "${SCP[@]}" "$HOST:$remote_components" "$component_file"
  "${SCP[@]}" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_final_components.npz" \
    "$component_archive"
  "${SCP[@]}" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_config.json" \
    "$HOST:/home/ubuntu/ecup/work/${remote_tag}_history.json" \
    "$ROOT/work/"
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/${val_tag}_best_val_components.npz"; then
    "${SCP[@]}" \
      "$HOST:/home/ubuntu/ecup/work/${val_tag}_best_val_components.npz" \
      "$ROOT/work/${val_tag}_server_best_val_components.npz"
  fi
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/${val_tag}_val.npy"; then
    "${SCP[@]}" \
      "$HOST:/home/ubuntu/ecup/work/${val_tag}_val.npy" \
      "$ROOT/work/${val_tag}_server_val.npy"
  fi
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "FETCHED $remote_tag (no submission built)"
}

fetch_candidate cls300_full
fetch_candidate cls409_full
fetch_candidate cls300cal_full
fetch_candidate cls300mkt_full

# The wide run has an automatic batch-size fallback and therefore one of two tags.
while true; do
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/cls409wide_full_final.npy"; then
    fetch_candidate cls409wide_full
    break
  fi
  if "${SSH[@]}" "test -s /home/ubuntu/ecup/work/cls409wide768_full_final.npy"; then
    fetch_candidate cls409wide768_full
    break
  fi
  sleep 30
done

fetch_candidate cls409class_full
fetch_candidate cls120_full
fetch_candidate cls300b_full
fetch_candidate cls60_full

echo "GPU_FETCH_DONE"
