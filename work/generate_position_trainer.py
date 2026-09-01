#!/usr/bin/env python
"""Mechanically derive the frozen relative-position trainer from its control."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "train_classifier_gpu_event_frozen.py"
OUTPUT = ROOT / "work" / "train_classifier_gpu_position_frozen.py"
EXPECTED_SOURCE_SHA256 = (
    "832ded891349db7d521bbb6f954ac9d0bc84f0986a4e1b56072dd2fe0f5e0a67"
)

source_bytes = SOURCE.read_bytes()
actual = hashlib.sha256(source_bytes).hexdigest()
if actual != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(f"control source drift: expected {EXPECTED_SOURCE_SHA256}, got {actual}")
source = source_bytes.decode()


def replace_once(old: str, new: str):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one source block, found {source.count(old)}")
    source = source.replace(old, new)


replace_once(
    '"""Corrected GPU multitask classifier with validation and full-retrain modes."""',
    '''"""Frozen TCN control plus explicit leakage-free relative-time channels.

Generated mechanically by ``generate_position_trainer.py`` from the exact
matched-control source.  The only model change is two deterministic channels:
linear relative age and a 30-day exponential recency decay.
"""''',
)
replace_once(
    """    input_channels = (
        nchannels
        + (nchannels if args.market else 0)
""",
    """    input_channels = (
        nchannels
        + 2  # linear relative age + 30-day exponential recency decay
        + (nchannels if args.market else 0)
""",
)
replace_once(
    """        output = output / scales
        if args.market:
""",
    """        output = output / scales
        ages = np.arange(args.window - 1, -1, -1, dtype=np.float32)
        position_values = np.stack([
            -ages / max(1, args.window - 1),
            np.exp(-ages / 30.0),
        ])
        if left < 0:
            position_values[:, :-left] = 0.0
        position_batch = np.broadcast_to(
            position_values[None, :, :],
            (len(index), 2, args.window),
        )
        output = np.concatenate([output, position_batch], axis=1)
        if args.market:
""",
)
replace_once(
    """        names
        + (["market_" + name for name in names] if args.market else [])
""",
    """        names
        + ["relative_age_linear", "relative_age_decay30"]
        + (["market_" + name for name in names] if args.market else [])
""",
)
replace_once(
    'config = vars(args) | {"channel_names": channel_names, "parameters": parameter_count}',
    'config = vars(args) | {"channel_names": channel_names, "parameters": parameter_count, "relative_position_channels": 2}',
)
replace_once(
    'f"event_summary={args.event_summary} "',
    'f"event_summary={args.event_summary} relative_position_channels=2 "',
)

OUTPUT.write_text(source)
print(f"generated {OUTPUT}")
print(f"sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
