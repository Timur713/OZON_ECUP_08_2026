#!/bin/bash
# waits for queue2 (tcn90, gru180) then trains the remaining diversity axis: window length
while pgrep -f "queue2.sh" >/dev/null; do sleep 30; done
sleep 10
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
S=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/train_seq2.py
$V $S tcn270  tcn 270 2 7 direct
$V $S tcn365b tcn 365 2 8 direct
$V $S tcn45   tcn 45  2 9 direct
