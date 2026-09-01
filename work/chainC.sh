#!/bin/bash
while pgrep -f exp_residcurve.py >/dev/null; do sleep 30; done
sleep 10
/Users/timur/Desktop/dev/OZON_ECUP_2026_3/.venv/bin/python \
  /Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/exp_seascond.py \
  > /Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/exp_seascond.log 2>&1
