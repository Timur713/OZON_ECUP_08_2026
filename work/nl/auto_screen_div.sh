#!/bin/bash
# Pull every finished diversity base and screen it against the admitted pool.
# Safe to run repeatedly; screening is cheap and idempotent.
set -u
ROOT="$1"; PY="$2"
cd "$ROOT"
mkdir -p work/div
rsync -az -e "ssh -F /tmp/sshcfg" --include='*_val.npy' --include='*_final.npy' \
      --include='*_history.json' --include='*_config.json' --exclude='*' \
      gpu:/home/ubuntu/ecup3/work/div/ work/div/ 2>/dev/null
ls work/div/*_val.npy > /dev/null 2>&1 || { echo "no bases yet"; exit 0; }
"$PY" work/nl/screen_candidates.py work/div/*_val.npy --out 173_screen_div.json 2>&1 | tail -40
