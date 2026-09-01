#!/bin/bash
while pgrep -f "train_seq2.py tcn180two" >/dev/null; do sleep 20; done
sleep 10
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
S=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/train_seq2.py
$V $S tcn90  tcn 90  2 6 direct
$V $S gru180 gru 180 2 5 direct
