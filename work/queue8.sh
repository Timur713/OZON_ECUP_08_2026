#!/bin/bash
while pgrep -f "queue7.sh" >/dev/null; do sleep 60; done
sleep 20
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
D=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work
# machine is alone now -> the 20-anchor GBDT fits without swap
$V $D/gbdt_dense262.py
# more seeds on the cheap short windows that proved useful in the stack
$V $D/train_seq2.py w45c  tcn 45  2 61 direct
$V $D/train_seq2.py w45d  tcn 45  2 62 direct
$V $D/train_seq2.py w60c  tcn 60  2 63 direct
$V $D/train_seq2.py w90c  tcn 90  2 64 direct
$V $D/train_seq2.py w120c tcn 120 2 65 direct
$V $D/train_seq2.py w409a tcn 409 2 66 direct
