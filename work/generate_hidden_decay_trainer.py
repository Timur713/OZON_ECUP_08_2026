#!/usr/bin/env python
"""Mechanically derive a frozen multi-scale hidden-pooling trainer."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "train_classifier_gpu_event_frozen.py"
OUTPUT = ROOT / "work" / "train_classifier_gpu_hidden_decay_frozen.py"
EXPECTED_SOURCE_SHA256 = (
    "832ded891349db7d521bbb6f954ac9d0bc84f0986a4e1b56072dd2fe0f5e0a67"
)

source_bytes = SOURCE.read_bytes()
actual = hashlib.sha256(source_bytes).hexdigest()
if actual != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(
        f"control source drift: expected {EXPECTED_SOURCE_SHA256}, got {actual}"
    )
source = source_bytes.decode()


def replace_once(old: str, new: str):
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one source block, found {count}")
    source = source.replace(old, new)


replace_once(
    '"""Corrected GPU multitask classifier with validation and full-retrain modes."""',
    '''"""Frozen TCN control plus multi-scale decay pooling of hidden states.

Generated mechanically by ``generate_hidden_decay_trainer.py`` from the exact
matched-control source.  The only representation change is four deterministic
recency-weighted averages of the learned temporal trunk.
"""''',
)
replace_once(
    "HORIZONS = (7, 14, 30, 60)\n",
    "HORIZONS = (7, 14, 30, 60)\nHIDDEN_DECAYS = (7.0, 30.0, 90.0, 180.0)\n",
)
replace_once(
    """        event_summary=False,
    ):
""",
    """        event_summary=False,
        hidden_decay_pooling=False,
    ):
""",
)
replace_once(
    """        self.event_summary = event_summary
        self.summary_windows""",
    """        self.event_summary = event_summary
        self.hidden_decay_pooling = hidden_decay_pooling
        self.summary_windows""",
)
replace_once(
    """                width * 3 + summary_width + event_width + static_features,
""",
    """                width * (3 + (len(HIDDEN_DECAYS) if hidden_decay_pooling else 0))
                + summary_width + event_width + static_features,
""",
)
replace_once(
    """        pooled_parts = [y.mean(-1), y.max(-1).values, y[..., -14:].mean(-1)]
        if self.multiwindow_summary:
""",
    """        pooled_parts = [y.mean(-1), y.max(-1).values, y[..., -14:].mean(-1)]
        if self.hidden_decay_pooling:
            ages = torch.arange(
                y.shape[-1] - 1, -1, -1, device=y.device, dtype=torch.float32
            )
            for decay in HIDDEN_DECAYS:
                weights = torch.softmax(-ages / decay, dim=0).to(y.dtype)
                pooled_parts.append((y * weights[None, None, :]).sum(-1))
        if self.multiwindow_summary:
""",
)
replace_once(
    """        event_summary=args.event_summary,
    ).to(device)
""",
    """        event_summary=args.event_summary,
        hidden_decay_pooling=True,
    ).to(device)
""",
)
replace_once(
    'f"event_summary={args.event_summary} "',
    'f"event_summary={args.event_summary} hidden_decay_pooling={HIDDEN_DECAYS} "',
)
replace_once(
    'config = vars(args) | {"channel_names": channel_names, "parameters": parameter_count}',
    'config = vars(args) | {"channel_names": channel_names, "parameters": parameter_count, "hidden_decay_pooling": list(HIDDEN_DECAYS)}',
)

OUTPUT.write_text(source)
print(f"generated {OUTPUT}")
print(f"sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
