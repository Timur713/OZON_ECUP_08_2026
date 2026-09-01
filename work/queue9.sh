#!/bin/bash
while pgrep -f "sweep336.py" >/dev/null; do sleep 30; done
sleep 15
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
D=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work
$V $D/gbdt_f4.py
$V $D/train_seq2.py a409a tcn 409 2 71 direct 12 378 all
$V $D/train_seq2.py a120a tcn 120 2 72 direct 12 378 all
$V $D/train_seq2.py a45a  tcn 45  2 73 direct 12 378 all
$V $D/train_seq2.py a409b tcn 409 2 74 direct 12 378 all
