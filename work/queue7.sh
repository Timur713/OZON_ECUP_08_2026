#!/bin/bash
while pgrep -f "queue6.sh" >/dev/null; do sleep 60; done
sleep 10
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
S=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/train_seq2.py
# fan of receptive fields, two seeds each -- cheap short windows first
$V $S w45a  tcn 45  2 41 direct
$V $S w45b  tcn 45  2 42 direct
$V $S w60a  tcn 60  2 43 direct
$V $S w60b  tcn 60  2 44 direct
$V $S w90a  tcn 90  2 45 direct
$V $S w90b  tcn 90  2 46 direct
$V $S w120a tcn 120 2 47 direct
$V $S w120b tcn 120 2 48 direct
$V $S w150a tcn 150 2 49 direct
$V $S w180a tcn 180 2 50 direct
$V $S w180b tcn 180 2 51 direct
$V $S w270a tcn 270 2 52 direct
$V $S w365a tcn 365 2 53 direct
$V $S w365b tcn 365 2 54 direct
