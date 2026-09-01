#!/bin/bash
while pgrep -f "train_seq2.py tcn365v336" >/dev/null; do sleep 30; done
sleep 10
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
D=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work
$V $D/train_seq2.py tcn409 tcn 409 2 11 direct
$V $D/gbdt_dense262.py
$V $D/train_seq2.py tcn365e tcn 365 2 31 direct
