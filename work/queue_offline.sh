#!/bin/bash
D=/Users/timur/Desktop/dev/OZON_ECUP_2026_3
V=$D/.venv/bin/python
cd $D
guard () {  # refuse to start if disk is低 or swap is already big
  free_gb=$(df -m /Users/timur | tail -1 | awk '{print int($4/1024)}')
  if [ "$free_gb" -lt 6 ]; then echo "ABORT: only ${free_gb}GB disk left"; exit 1; fi
}
while pgrep -f '[g]bdt_surv.py' >/dev/null; do sleep 60; done
guard; echo "=== gbdt_seasw (season-weighted anchors) ==="; date
$V $D/work/gbdt_seasw.py > $D/work/gbdt_seasw.log 2>&1
guard; echo "=== tcn window 210, unexplored length ==="; date
$V $D/work/train_seq2.py w210a tcn 210 2 91 direct >> $D/work/offline.log 2>&1
guard; echo "=== tcn window 300 ==="; date
$V $D/work/train_seq2.py w300a tcn 300 2 92 direct >> $D/work/offline.log 2>&1
guard; echo "=== tcn 409, extra seed ==="; date
$V $D/work/train_seq2.py w409c tcn 409 2 93 direct >> $D/work/offline.log 2>&1
echo "=== QUEUE DONE ==="; date
