#!/bin/bash
while pgrep -f "queue9.sh" >/dev/null; do sleep 60; done
sleep 15
V=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python
D=/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work
# moved to parallel run
$V $D/gbdt_surv.py
