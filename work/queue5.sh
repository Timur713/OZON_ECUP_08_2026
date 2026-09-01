#!/bin/bash
while pgrep -f "queue4.sh" >/dev/null; do sleep 30; done
sleep 10
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
S=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/train_seq2.py
# denser anchors: 3x the training data per epoch, zero extra memory
$V $S tcn365d4 tcn 365 1 14 direct 4
$V $S tcn365d6 tcn 365 2 15 direct 6
