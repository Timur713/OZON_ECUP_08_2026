#!/usr/bin/env python
"""Frozen marked-event profile layered on the regularity trainer.

The network shape remains identical to the six-value event-profile branch.
Only the four profile values after exact recency/latest-gap are replaced by
bounded statistics of the values attached to the last five nonzero events.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


DEPENDENCY = Path(__file__).with_name("train_classifier_gpu_regularity_frozen.py")
spec = importlib.util.spec_from_file_location("ecup_regularity_frozen", DEPENDENCY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen dependency {DEPENDENCY}")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def marked_event_profile(raw):
    """Return recency/gap plus bounded recent event-value statistics."""
    recency, last_gap = core.event_cadence(raw)
    positions = (
        torch.ones(1, device=raw.device, dtype=raw.dtype)
        if raw.shape[-1] == 1 else
        torch.linspace(0.0, 1.0, raw.shape[-1], device=raw.device, dtype=raw.dtype)
    )
    event_positions = torch.where(
        raw > 0,
        positions[None, None, :],
        torch.full((), -1.0, device=raw.device, dtype=raw.dtype),
    )
    recent_positions, recent_indices = torch.topk(
        event_positions, k=min(5, raw.shape[-1]), dim=-1
    )
    recent_values = torch.gather(raw, -1, recent_indices)
    valid = recent_positions >= 0
    valid_float = valid.to(raw.dtype)
    count = valid_float.sum(-1)
    safe_count = count.clamp_min(1.0)
    mean_raw = (recent_values * valid_float).sum(-1) / safe_count
    last_raw = torch.where(
        valid[..., 0], recent_values[..., 0], torch.zeros_like(mean_raw)
    )
    centered = (recent_values - mean_raw[..., None]) * valid_float
    variance = centered.square().sum(-1) / safe_count
    value_cv = torch.sqrt(variance.clamp_min(0.0)) / (mean_raw + 1e-4)
    last_to_mean = last_raw / (mean_raw + 1e-4)
    bounded_last = last_raw.clamp_min(0.0) / (1.0 + last_raw.clamp_min(0.0))
    bounded_mean = mean_raw.clamp_min(0.0) / (1.0 + mean_raw.clamp_min(0.0))
    bounded_cv = value_cv.clamp(0.0, 4.0) / 4.0
    bounded_ratio = last_to_mean.clamp(0.0, 4.0) / 4.0
    return (
        recency,
        last_gap,
        bounded_last,
        bounded_mean,
        bounded_cv,
        bounded_ratio,
    )


core.event_regularity_profile = marked_event_profile

if __name__ == "__main__":
    core.main()
