#!/usr/bin/env python
"""Frozen TCN control plus multi-scale decay pooling of hidden states.

Generated mechanically by ``generate_hidden_decay_trainer.py`` from the exact
matched-control source.  The only representation change is four deterministic
recency-weighted averages of the learned temporal trunk.
"""
import argparse
import datetime as dt
import gc
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MD = os.environ.get("ECUP_MAT", "/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat/")
OUT = os.environ.get("ECUP_OUT", "/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/")
HORIZONS = (7, 14, 30, 60)
HIDDEN_DECAYS = (7.0, 30.0, 90.0, 180.0)
SUMMARY_WINDOWS = (7, 14, 30, 60, 120, 180, 300, 409)
START_DATE = dt.date(2025, 1, 1)
CALENDAR_NAMES = [
    "target_mid_sin1", "target_mid_cos1",
    "target_mid_sin2", "target_mid_cos2",
    "target_mid_sin3", "target_mid_cos3",
    "absolute_time", "contains_feb23", "contains_mar08", "contains_newyear",
]
TARGET_PROFILE_CHANNELS = ("gmv", "to_ord", "active", "searches")
TARGET_PROFILE_STATS = ("mean", "std", "max", "early", "late", "late_minus_early")
TARGET_PROFILE_NAMES = [
    f"target_profile_{channel}_{stat}"
    for channel in TARGET_PROFILE_CHANNELS
    for stat in TARGET_PROFILE_STATS
]
CHANNELS = {
    "base": ["gmv", "to_ord", "to_cart", "searches", "active", "search", "cat"],
    "all": [
        "gmv", "to_ord", "to_cart", "searches", "active", "search", "cat",
        "gmv_search", "gmv_cat", "search_to_ord", "cat_to_ord",
        "has_search_to_ord", "has_search_to_cart", "has_cat_to_ord", "has_cat_to_cart",
        "search_to_cart", "cat_to_cart",
    ],
}


def device_name():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def event_cadence(raw):
    """Return normalized recency and latest inter-event gap per channel."""
    positions = (
        torch.ones(1, device=raw.device, dtype=raw.dtype)
        if raw.shape[-1] == 1 else
        torch.linspace(
            0.0, 1.0, raw.shape[-1], device=raw.device, dtype=raw.dtype
        )
    )
    event_positions = torch.where(
        raw > 0,
        positions[None, None, :],
        torch.full((), -1.0, device=raw.device, dtype=raw.dtype),
    )
    latest_values = torch.topk(
        event_positions, k=min(2, raw.shape[-1]), dim=-1
    ).values
    latest = latest_values[..., 0]
    previous = (
        latest_values[..., 1]
        if raw.shape[-1] > 1 else torch.full_like(latest, -1.0)
    )
    recency = torch.where(latest >= 0, 1.0 - latest, 1.0)
    last_gap = torch.where(previous >= 0, latest - previous, 1.0)
    return recency, last_gap


class Network(nn.Module):
    def __init__(
        self, channels, width, blocks, length, summary_channels=0,
        static_features=0, survival_head=False, multiwindow_summary=False,
        event_summary=False,
        hidden_decay_pooling=False,
    ):
        super().__init__()
        self.summary_channels = summary_channels
        self.static_features = static_features
        self.survival_head = survival_head
        self.multiwindow_summary = multiwindow_summary
        self.event_summary = event_summary
        self.hidden_decay_pooling = hidden_decay_pooling
        self.summary_windows = tuple(sorted({min(value, length) for value in SUMMARY_WINDOWS}))
        self.input = nn.Conv1d(channels, width, 5, padding=2)
        self.blocks = nn.ModuleList()
        dilation = 1
        for _ in range(blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.BatchNorm1d(width),
            ))
            dilation = min(dilation * 2, max(1, length // 4))
        summary_width = (
            summary_channels * len(self.summary_windows) * 2
            if multiwindow_summary else 0
        )
        event_width = summary_channels * 2 if event_summary else 0
        self.trunk = nn.Sequential(
            nn.Linear(
                width * (3 + (len(HIDDEN_DECAYS) if hidden_decay_pooling else 0))
                + summary_width + event_width + static_features,
                512,
            ),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
        )
        self.classifier = nn.Linear(256, len(HORIZONS))
        if survival_head:
            # Start near an 18% 30-day event rate instead of the 87.5% implied
            # by four zero-logit softplus increments.  This avoids spending an
            # epoch merely undoing a pathological initial cumulative hazard.
            interval_days = torch.tensor([7.0, 7.0, 16.0, 30.0])
            initial_hazard = interval_days * 0.007
            initial_bias = torch.log(torch.expm1(initial_hazard))
            nn.init.zeros_(self.classifier.weight)
            with torch.no_grad():
                self.classifier.bias.copy_(initial_bias)
        self.magnitude = nn.Linear(256, 1)
        self.direct = nn.Linear(256, 1)

    def forward(self, x, static=None):
        y = torch.relu(self.input(x))
        for block in self.blocks:
            y = y + block(y)
        pooled_parts = [y.mean(-1), y.max(-1).values, y[..., -14:].mean(-1)]
        if self.hidden_decay_pooling:
            ages = torch.arange(
                y.shape[-1] - 1, -1, -1, device=y.device, dtype=torch.float32
            )
            for decay in HIDDEN_DECAYS:
                weights = torch.softmax(-ages / decay, dim=0).to(y.dtype)
                pooled_parts.append((y * weights[None, None, :]).sum(-1))
        if self.multiwindow_summary:
            raw = x[:, :self.summary_channels]
            for horizon in self.summary_windows:
                recent = raw[..., -horizon:]
                pooled_parts.extend([
                    recent.mean(-1),
                    (recent > 0).to(recent.dtype).mean(-1),
                ])
        if self.event_summary:
            raw = x[:, :self.summary_channels]
            # Exact event cadence is awkward for a pooled convolutional trunk
            # to recover.  Two bounded, scale-free values per raw channel make
            # it explicit: time since the latest event and the latest
            # inter-event gap.  Missing/one-event histories map to 1.0 rather
            # than introducing NaNs or a learned sentinel.
            recency, last_gap = event_cadence(raw)
            pooled_parts.extend([recency, last_gap])
        if self.static_features:
            if static is None or static.shape[1] != self.static_features:
                raise ValueError("invalid static-feature batch")
            pooled_parts.append(static)
        pooled = torch.cat(pooled_parts, dim=1)
        trunk = self.trunk(pooled)
        logits = self.classifier(trunk)
        if self.survival_head:
            # Non-negative hazard increments for (0,7], (7,14], (14,30],
            # and (30,60] make cumulative purchase probabilities monotone.
            # Convert cumulative hazards to numerically stable logits so the
            # existing BCE objective and inference code remain unchanged.
            cumulative_hazard = torch.cumsum(F.softplus(logits), dim=1).float()
            cumulative_hazard = cumulative_hazard.clamp_min(1e-6)
            logits = cumulative_hazard + torch.log(
                -torch.expm1(-cumulative_hazard)
            )
        magnitude = F.softplus(self.magnitude(trunk).squeeze(-1))
        direct = self.direct(trunk).squeeze(-1)
        return logits, magnitude, direct


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--mode", choices=("validate", "final"), default="validate")
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--channels", choices=tuple(CHANNELS), default="all")
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--pred-bs", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--mix", type=float, default=0.7,
                        help="weight of hurdle prediction versus direct regression")
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--magnitude-weight", type=float, default=0.5)
    parser.add_argument("--direct-weight", type=float, default=0.5)
    parser.add_argument("--val", type=int, default=378)
    parser.add_argument("--anchor-start", type=int, default=60)
    parser.add_argument("--special-anchor", type=int, default=-1)
    parser.add_argument("--special-repeat", type=int, default=1)
    parser.add_argument(
        "--user-holdout-anchor", type=int, default=-1,
        help="score validation on a fixed historical-season anchor",
    )
    parser.add_argument(
        "--user-holdout-frac", type=float, default=0.0,
        help=(
            "in validate mode, exclude this fraction of users from every "
            "training anchor and score them at --user-holdout-anchor"
        ),
    )
    parser.add_argument("--calendar", action="store_true")
    parser.add_argument(
        "--market", action="store_true",
        help="append standardized aggregate-market history for every raw channel",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="append explicit multi-window means and nonzero frequencies to the trunk",
    )
    parser.add_argument(
        "--event-summary", action="store_true",
        help=(
            "append exact per-channel recency and latest inter-event gap to "
            "the pooled trunk"
        ),
    )
    parser.add_argument(
        "--survival-head", action="store_true",
        help="use monotone cumulative-hazard logits for 7/14/30/60-day purchase",
    )
    parser.add_argument(
        "--target-profile",
        action="store_true",
        help=(
            "append aggregate target-season covariates; validation/final use "
            "the same calendar window one year earlier"
        ),
    )
    parser.add_argument(
        "--target-profile-head", action="store_true",
        help="feed target-profile covariates once to the pooled head, not Conv1d",
    )
    parser.add_argument("--nusers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--private-selection", action="store_true",
        help=(
            "fit validation calibration on a fixed 20% user split and score "
            "epoch/head mix on the independent 80%"
        ),
    )
    return parser.parse_args()


def main():
    args = arguments()
    if not 0 < args.frac <= 1:
        raise ValueError("--frac must be in (0, 1]")
    if not 0 <= args.mix <= 1:
        raise ValueError("--mix must be in [0, 1]")
    if min(args.class_weight, args.magnitude_weight, args.direct_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if args.anchor_start < 0:
        raise ValueError("--anchor-start must be non-negative")
    if args.special_repeat < 1:
        raise ValueError("--special-repeat must be at least 1")
    if not 0 <= args.user_holdout_frac < 1:
        raise ValueError("--user-holdout-frac must be in [0, 1)")
    if args.user_holdout_frac and args.user_holdout_anchor < 0:
        raise ValueError("--user-holdout-anchor is required for a user holdout")
    if args.target_profile_head and not args.target_profile:
        raise ValueError("--target-profile-head requires --target-profile")
    os.makedirs(OUT, exist_ok=True)
    device = device_name()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    names = CHANNELS[args.channels]
    arrays = []
    for name in names:
        values = np.load(os.path.join(MD, name + ".npy"), mmap_mode="r")
        if args.nusers:
            values = values[:args.nusers]
        values = np.array(values, dtype=np.float32, copy=True)
        if name != "active" and not name.startswith("has_"):
            np.log1p(values, out=values)
        arrays.append(values.astype(np.float16))
    matrix = np.stack(arrays, axis=1)
    del arrays
    gc.collect()
    nusers, nchannels, ndays = matrix.shape
    final_anchor = ndays - 1
    cutoff = args.val if args.mode == "validate" else final_anchor
    if args.mode == "validate" and args.val + 30 > final_anchor:
        raise ValueError("validation target ends outside observed data")

    holdout_active = args.mode == "validate" and args.user_holdout_frac > 0
    training_users = np.arange(nusers)
    validation_users = None
    validation_anchor = args.val
    if holdout_active:
        if args.user_holdout_anchor + 30 > final_anchor:
            raise ValueError("user-holdout target ends outside observed data")
        split_rng = np.random.default_rng(args.seed + 99173)
        holdout_size = max(1, int(round(nusers * args.user_holdout_frac)))
        validation_users = np.sort(
            split_rng.choice(nusers, holdout_size, replace=False)
        )
        training_mask = np.ones(nusers, dtype=bool)
        training_mask[validation_users] = False
        training_users = np.flatnonzero(training_mask)
        validation_anchor = args.user_holdout_anchor
        np.save(os.path.join(OUT, args.tag + "_val_users.npy"), validation_users)

    sample = matrix[training_users[::37]].astype(np.float32)
    scales = (sample.std(axis=(0, 2)) + 1e-6).astype(np.float32)[None, :, None]
    del sample
    population = None
    market = None
    if args.market or args.target_profile:
        if holdout_active:
            population = np.zeros((nchannels, ndays), dtype=np.float64)
            for start in range(0, len(training_users), 8192):
                index = training_users[start:start + 8192]
                population += matrix[index].sum(axis=0, dtype=np.float64)
            population = (population / len(training_users)).astype(np.float32)
        else:
            population = matrix.mean(axis=0, dtype=np.float32)
    if args.market:
        # Daily population aggregates tell the network which part of a user's
        # recent history is idiosyncratic and which part is a market-wide
        # seasonal shock.  Normalize only on the observable prefix so that the
        # validation target window cannot influence its scale.
        market = population
        market_prefix = market[:, :cutoff + 1]
        market_mean = market_prefix.mean(axis=1, keepdims=True)
        market_std = market_prefix.std(axis=1, keepdims=True) + 1e-6
        market = ((market - market_mean) / market_std).astype(np.float32)
    gmv = np.load(os.path.join(MD, "gmv.npy"), mmap_mode="r")[:nusers]
    cumulative = np.zeros((nusers, ndays + 1), dtype=np.float64)
    np.cumsum(gmv, axis=1, out=cumulative[:, 1:])
    del gmv

    def target(anchor, horizon):
        end = anchor + horizon
        if end > final_anchor:
            raise ValueError(f"target {anchor=} {horizon=} is unavailable")
        return np.log1p(cumulative[:, end + 1] - cumulative[:, anchor + 1]).astype(np.float32)

    anchors = list(range(args.anchor_start, cutoff - min(HORIZONS) + 1, args.stride))
    latest_anchor = cutoff - min(HORIZONS)
    if latest_anchor not in anchors:
        anchors.append(latest_anchor)
        anchors.sort()
    if args.special_anchor >= 0 and args.special_repeat > 1:
        if args.special_anchor not in anchors:
            raise ValueError("--special-anchor must belong to the generated anchor grid")
        anchors.extend([args.special_anchor] * (args.special_repeat - 1))
    available = {t: tuple(h for h in HORIZONS if t + h <= cutoff) for t in anchors}
    labels = {t: {h: target(t, h) for h in available[t]} for t in anchors}
    validation_target = (
        target(validation_anchor, 30)[validation_users]
        if holdout_active
        else (target(args.val, 30) if args.mode == "validate" else None)
    )

    training_sample_size = min(int(nusers * args.frac), len(training_users))
    batches = training_sample_size // args.bs
    if batches < 1:
        raise ValueError("sample contains less than one batch")
    planned_steps = args.epochs * len(anchors) * batches
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    input_channels = (
        nchannels
        + (nchannels if args.market else 0)
        + (len(CALENDAR_NAMES) if args.calendar else 0)
        + (
            len(TARGET_PROFILE_NAMES)
            if args.target_profile and not args.target_profile_head else 0
        )
    )
    network = Network(
        input_channels,
        args.width,
        args.blocks,
        args.window,
        summary_channels=(
            nchannels if args.summary or args.event_summary else 0
        ),
        static_features=(
            len(TARGET_PROFILE_NAMES) if args.target_profile_head else 0
        ),
        survival_head=args.survival_head,
        multiwindow_summary=args.summary,
        event_summary=args.event_summary,
        hidden_decay_pooling=True,
    ).to(device)
    parameter_count = sum(p.numel() for p in network.parameters())
    print(
        f"{args.tag}: mode={args.mode} device={device} L={args.window} width={args.width} "
        f"blocks={args.blocks} channels={input_channels} raw_channels={nchannels} "
        f"calendar={args.calendar} market={args.market} summary={args.summary} "
        f"event_summary={args.event_summary} hidden_decay_pooling={HIDDEN_DECAYS} "
        f"survival_head={args.survival_head} "
        f"target_profile={args.target_profile} "
        f"anchors={len(anchors)} users={nusers} train_users={len(training_users)} "
        f"validation_users={0 if validation_users is None else len(validation_users)} "
        f"parameters={parameter_count/1e6:.2f}M matrix={matrix.nbytes/1e9:.2f}GB steps={total_steps}",
        flush=True,
    )
    print(f"channel_scales={np.round(scales.ravel(), 4).tolist()}", flush=True)

    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1), pct_start=0.15
    )
    amp = device == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    binary_loss = nn.BCEWithLogitsLoss()

    def calendar_vector(anchor):
        target_start = START_DATE + dt.timedelta(days=anchor + 1)
        target_end = target_start + dt.timedelta(days=29)
        target_mid = target_start + dt.timedelta(days=14)
        phase = 2 * math.pi * (target_mid.timetuple().tm_yday - 1) / 365.25
        days = {target_start + dt.timedelta(days=offset) for offset in range(30)}
        contains = lambda month, day: any(value.month == month and value.day == day for value in days)
        return np.array([
            math.sin(phase), math.cos(phase),
            math.sin(2 * phase), math.cos(2 * phase),
            math.sin(3 * phase), math.cos(3 * phase),
            (anchor - final_anchor / 2) / final_anchor,
            float(contains(2, 23)), float(contains(3, 8)),
            float(any((value.month == 1 and value.day <= 8) or
                      (value.month == 12 and value.day == 31) for value in days)),
        ], dtype=np.float32)

    calendar_values = {
        anchor: calendar_vector(anchor)
        for anchor in set(anchors + [args.val, final_anchor])
    } if args.calendar else {}

    target_profile_values = {}
    if args.target_profile:
        channel_index = {name: names.index(name) for name in TARGET_PROFILE_CHANNELS}
        profile_rows = np.stack([
            population[channel_index[name]] for name in TARGET_PROFILE_CHANNELS
        ]).astype(np.float32)
        profile_prefix = profile_rows[:, :cutoff + 1]
        profile_mean = profile_prefix.mean(axis=1, keepdims=True)
        profile_std = profile_prefix.std(axis=1, keepdims=True) + 1e-6
        profile_rows = (profile_rows - profile_mean) / profile_std

        def target_profile_vector(anchor):
            source_start = anchor + 1
            # Training anchors use known historical seasons. Validation and
            # final inference use the same calendar dates one year earlier, so
            # their target labels never enter these features.
            if source_start + 30 > cutoff + 1:
                source_start -= 365
            while source_start + 30 > ndays:
                source_start -= 365
            if source_start < 0:
                # The first validation target starts 13 days before an exact
                # previous-year proxy exists in the observable history. Use
                # the earliest complete 30-day window instead; this remains
                # leakage-free and is preferable to dropping the validation
                # anchor that selects the final epoch.
                source_start = 0
            window = profile_rows[:, source_start:source_start + 30]
            if window.shape[1] != 30:
                raise ValueError(f"invalid target-profile window for anchor {anchor}")
            early = window[:, :15].mean(axis=1)
            late = window[:, 15:].mean(axis=1)
            stats = np.stack([
                window.mean(axis=1),
                window.std(axis=1),
                window.max(axis=1),
                early,
                late,
                late - early,
            ], axis=1)
            return stats.reshape(-1).astype(np.float32)

        target_profile_values = {
            anchor: target_profile_vector(anchor)
            for anchor in set(anchors + [args.val, final_anchor])
        }

    def input_window(index, anchor):
        left = anchor - args.window + 1
        if left >= 0:
            output = matrix[index, :, left:anchor + 1].astype(np.float32)
        else:
            output = np.zeros((len(index), nchannels, args.window), dtype=np.float32)
            output[:, :, -(anchor + 1):] = matrix[index, :, :anchor + 1].astype(np.float32)
        output = output / scales
        if args.market:
            market_output = np.zeros((nchannels, args.window), dtype=np.float32)
            source_left = max(left, 0)
            destination_left = source_left - left
            market_output[:, destination_left:] = market[:, source_left:anchor + 1]
            market_batch = np.broadcast_to(
                market_output[None, :, :],
                (len(index), nchannels, args.window),
            )
            output = np.concatenate([output, market_batch], axis=1)
        if args.calendar:
            calendar = np.broadcast_to(
                calendar_values[anchor][None, :, None],
                (len(index), len(CALENDAR_NAMES), args.window),
            )
            output = np.concatenate([output, calendar], axis=1)
        if args.target_profile:
            target_profile = np.broadcast_to(
                target_profile_values[anchor][None, :, None],
                (len(index), len(TARGET_PROFILE_NAMES), args.window),
            )
            if not args.target_profile_head:
                output = np.concatenate([output, target_profile], axis=1)
        return output

    def static_batch(index, anchor):
        if not args.target_profile_head:
            return None
        values = np.broadcast_to(
            target_profile_values[anchor][None, :],
            (len(index), len(TARGET_PROFILE_NAMES)),
        ).copy()
        return torch.from_numpy(values).to(device, non_blocking=True)

    def predict(anchor, components_path=None, indices=None):
        network.eval()
        prediction_users = np.arange(nusers) if indices is None else np.asarray(indices)
        prediction_size = len(prediction_users)
        result = np.empty(prediction_size, dtype=np.float32)
        hurdle_result = np.empty(prediction_size, dtype=np.float32) if components_path else None
        direct_result = np.empty(prediction_size, dtype=np.float32) if components_path else None
        probability_result = np.empty(prediction_size, dtype=np.float32) if components_path else None
        probability_horizons_result = (
            np.empty((prediction_size, len(HORIZONS)), dtype=np.float32)
            if components_path else None
        )
        magnitude_result = np.empty(prediction_size, dtype=np.float32) if components_path else None
        with torch.no_grad():
            for start in range(0, prediction_size, args.pred_bs):
                stop = min(start + args.pred_bs, prediction_size)
                index = prediction_users[start:stop]
                x = torch.from_numpy(input_window(index, anchor)).to(device, non_blocking=True)
                static = static_batch(index, anchor)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits, magnitude, direct = network(x, static)
                    probability_horizons = torch.sigmoid(logits)
                    probability = probability_horizons[:, HORIZONS.index(30)]
                    hurdle = probability * magnitude
                    output = args.mix * hurdle + (1.0 - args.mix) * direct
                tensors = {
                    "combined": output,
                    "hurdle": hurdle,
                    "direct": direct,
                    "probability": probability,
                    "magnitude": magnitude,
                }
                for component_name, component_value in tensors.items():
                    if not torch.isfinite(component_value).all():
                        bad = int((~torch.isfinite(component_value)).sum().item())
                        raise FloatingPointError(
                            f"non-finite {component_name} prediction at "
                            f"anchor={anchor} users={start}:{stop} count={bad}"
                        )
                if not torch.isfinite(probability_horizons).all():
                    raise FloatingPointError(
                        f"non-finite horizon probabilities at anchor={anchor} "
                        f"users={start}:{stop}"
                    )
                result[start:stop] = output.float().cpu().numpy()
                if components_path:
                    hurdle_result[start:stop] = hurdle.float().cpu().numpy()
                    direct_result[start:stop] = direct.float().cpu().numpy()
                    probability_result[start:stop] = probability.float().cpu().numpy()
                    probability_horizons_result[start:stop] = (
                        probability_horizons.float().cpu().numpy()
                    )
                    magnitude_result[start:stop] = magnitude.float().cpu().numpy()
        if components_path:
            np.savez(
                components_path,
                combined=result,
                hurdle=hurdle_result,
                direct=direct_result,
                probability=probability_result,
                probability_horizons=probability_horizons_result,
                magnitude=magnitude_result,
            )
            component_stem = components_path[:-4] if components_path.endswith(".npz") else components_path
            np.save(component_stem + "_hurdle.npy", hurdle_result)
        network.train()
        return result

    validation_public = None
    validation_private = None
    if args.private_selection:
        if args.mode != "validate" or holdout_active:
            raise ValueError(
                "--private-selection requires ordinary validation mode"
            )
        split_rng = np.random.default_rng(20260825)
        public_size = min(50_000, max(1, len(validation_target) // 5))
        validation_public = np.sort(
            split_rng.choice(len(validation_target), public_size, replace=False)
        )
        private_mask = np.ones(len(validation_target), dtype=bool)
        private_mask[validation_public] = False
        validation_private = np.flatnonzero(private_mask)

    def calibrated_score(prediction, truth):
        if not np.isfinite(prediction).all():
            raise FloatingPointError("non-finite validation prediction")
        design = np.vstack([prediction, np.ones_like(prediction)]).T
        fit_index = (
            validation_public
            if validation_public is not None
            else np.arange(len(truth))
        )
        score_index = (
            validation_private
            if validation_private is not None
            else np.arange(len(truth))
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design[fit_index], truth[fit_index], rcond=None
        )
        calibrated = np.clip(design[score_index] @ coefficients, 0, None)
        score = float(
            np.sqrt(np.mean((truth[score_index] - calibrated) ** 2))
        )
        return score, coefficients

    channel_names = (
        names
        + (["market_" + name for name in names] if args.market else [])
        + (CALENDAR_NAMES if args.calendar else [])
        + (TARGET_PROFILE_NAMES if args.target_profile else [])
    )
    config = vars(args) | {"channel_names": channel_names, "parameters": parameter_count, "hidden_decay_pooling": list(HIDDEN_DECAYS)}
    with open(os.path.join(OUT, args.tag + "_config.json"), "w") as stream:
        json.dump(config, stream, indent=2)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    step = 0
    best = float("inf")
    history = []
    stopped = False
    for epoch in range(1, args.epochs + 1):
        for anchor_value in rng.permutation(anchors):
            anchor = int(anchor_value)
            users = rng.permutation(training_users)[:training_sample_size]
            for start in range(0, len(users) - args.bs + 1, args.bs):
                index = np.sort(users[start:start + args.bs])
                x = torch.from_numpy(input_window(index, anchor)).to(device, non_blocking=True)
                static = static_batch(index, anchor)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits, magnitude, direct = network(x, static)
                    classification = []
                    for head, horizon in enumerate(HORIZONS):
                        if horizon in available[anchor]:
                            z_h = labels[anchor][horizon][index]
                            binary = torch.from_numpy((z_h > 0).astype(np.float32)).to(device)
                            classification.append(binary_loss(logits[:, head], binary))
                    loss_classification = torch.stack(classification).mean()
                    loss = args.class_weight * loss_classification
                    if 30 in available[anchor]:
                        z30 = torch.from_numpy(labels[anchor][30][index]).to(device)
                        positive = z30 > 0
                        loss_magnitude = F.mse_loss(magnitude[positive], z30[positive])
                        loss_direct = F.mse_loss(direct, z30)
                        loss = (
                            args.class_weight * loss_classification
                            + args.magnitude_weight * loss_magnitude
                            + args.direct_weight * loss_direct
                        )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                optimizer.zero_grad(set_to_none=True)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                old_scale = amp_scaler.get_scale()
                amp_scaler.step(optimizer)
                amp_scaler.update()
                if amp_scaler.get_scale() >= old_scale:
                    scheduler.step()
                step += 1
                if step == 1 or step % 250 == 0:
                    vram = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
                    print(
                        f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.5f} "
                        f"elapsed={time.time()-started:.0f}s peak_vram={vram:.2f}GiB",
                        flush=True,
                    )
                if args.max_steps and step >= args.max_steps:
                    stopped = True
                    break
            if stopped:
                break

        checkpoint = {"model": network.state_dict(), "epoch": epoch, "config": config, "scales": scales.ravel()}
        torch.save(checkpoint, os.path.join(OUT, f"{args.tag}_epoch{epoch}.pt"))
        prediction_anchor = validation_anchor if args.mode == "validate" else final_anchor
        prediction_users = validation_users if holdout_active else None
        component_file = os.path.join(OUT, f"{args.tag}_{args.mode}_components.npz")
        prediction = predict(prediction_anchor, component_file, prediction_users)
        np.save(os.path.join(OUT, f"{args.tag}_{args.mode}_epoch{epoch}.npy"), prediction)
        shutil.copyfile(
            component_file,
            os.path.join(
                OUT, f"{args.tag}_{args.mode}_epoch{epoch}_components.npz"
            ),
        )
        if args.mode == "validate":
            with np.load(component_file) as component_values:
                hurdle = component_values["hurdle"].astype(np.float64)
                direct = component_values["direct"].astype(np.float64)
                component_scores = {}
                for key in ("combined", "hurdle", "direct", "probability", "magnitude"):
                    value_score, _ = calibrated_score(
                        component_values[key], validation_target
                    )
                    component_scores[key] = value_score
            mix_rows = []
            for hurdle_weight in np.linspace(0.0, 1.0, 21):
                mixed = hurdle_weight * hurdle + (1.0 - hurdle_weight) * direct
                mixed_score, mixed_coefficients = calibrated_score(
                    mixed, validation_target
                )
                mix_rows.append((mixed_score, float(hurdle_weight), mixed_coefficients))
            score, best_hurdle_weight, coefficients = min(mix_rows, key=lambda row: row[0])
            selected_prediction = (
                best_hurdle_weight * hurdle + (1.0 - best_hurdle_weight) * direct
            ).astype(np.float32)
            history.append({
                "epoch": epoch,
                "score": score,
                "combined_score": component_scores["combined"],
                "component_scores": component_scores,
                "best_hurdle_weight": best_hurdle_weight,
                "slope": float(coefficients[0]),
                "intercept": float(coefficients[1]),
            })
            print(
                f"EPOCH {epoch}: val_cal={score:.6f} "
                f"hurdle_weight={best_hurdle_weight:.2f} "
                f"combined={component_scores['combined']:.6f} "
                f"slope={coefficients[0]:.5f}",
                flush=True,
            )
            if score < best:
                best = score
                np.save(os.path.join(OUT, args.tag + "_val.npy"), selected_prediction)
                shutil.copyfile(
                    component_file,
                    os.path.join(OUT, args.tag + "_best_val_components.npz"),
                )
                best_final_components = os.path.join(
                    OUT, args.tag + "_best_final_components.npz"
                )
                predict(final_anchor, best_final_components)
                with np.load(best_final_components) as final_values:
                    selected_final = (
                        best_hurdle_weight * final_values["hurdle"]
                        + (1.0 - best_hurdle_weight) * final_values["direct"]
                    )
                np.save(os.path.join(OUT, args.tag + "_final.npy"), selected_final)
                torch.save(checkpoint, os.path.join(OUT, args.tag + "_best.pt"))
                print("  ^ saved new best", flush=True)
        else:
            history.append({"epoch": epoch})
            np.save(os.path.join(OUT, args.tag + "_final.npy"), prediction)
            torch.save(checkpoint, os.path.join(OUT, args.tag + "_best.pt"))
            print(f"EPOCH {epoch}: saved final prediction", flush=True)
        with open(os.path.join(OUT, args.tag + "_history.json"), "w") as stream:
            json.dump(history, stream, indent=2)
        if stopped:
            print("Stopped at smoke-test step limit", flush=True)
            break
    print(f"DONE {args.tag}: steps={step} elapsed={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
