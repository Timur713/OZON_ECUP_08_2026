#!/bin/bash
while pgrep -f "gbdt_seasw.py" >/dev/null; do sleep 30; done
sleep 10
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
S=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/train_seq2.py
# 10x the temporal diversity at the SAME number of gradient steps
$V $S d409 tcn 409 2 81 direct 2 378 base 0.10
$V $S d120 tcn 120 2 82 direct 2 378 base 0.10
$V $S d45  tcn 45  2 83 direct 2 378 base 0.10
