#!/usr/bin/env python
"""Frozen exact-w409c structural ablations.

This intentionally keeps the successful w409c recipe fixed: seven base
channels, width 96, recent anchors 186..348 with stride 12, full-user batches,
direct log-GMV loss, two epochs and the original scalar normalization.  The
The selected variant makes exactly one structural change: multiscale hidden
pooling, explicit relative-position input, or exact event-cadence pooling.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from datetime import date, timedelta

import numpy as np
import torch
import torch.nn as nn


CHANNELS = ["gmv", "to_ord", "to_cart", "searches", "active", "search", "cat"]
DECAYS = (7.0, 30.0, 90.0, 180.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nusers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--prediction-batch-size", type=int, default=4096)
    parser.add_argument("--anchor-start", type=int, default=186)
    parser.add_argument("--anchor-stride", type=int, default=12)
    parser.add_argument(
        "--anchor-end", type=int, default=0,
        help="last anchor to train on; 0 means the usual validation-safe limit",
    )
    parser.add_argument("--anchor-phase", type=int, default=0)
    parser.add_argument("--user-fraction", type=float, default=1.0)
    parser.add_argument("--user-seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument(
        "--mask-recent", type=int, default=0,
        help=(
            "zero the last N days of every input window. sw28 kept ONLY the "
            "recent weeks and was worth a measured +0.00011; this is the "
            "opposite restriction, forcing the network onto long-run structure "
            "so its errors cannot resemble those of models dominated by recent "
            "activity."
        ),
    )
    parser.add_argument(
        "--objective", default="direct",
        choices=("direct", "probability", "magnitude"),
        help=(
            "what the network is trained to predict. direct is the pool's "
            "recipe, the log of the 30-day sum. probability trains only the "
            "buy event, magnitude trains only on users who did buy. The pool "
            "contains hurdle HEADS inside one network and separate cls "
            "classifiers on the ordinary representation, but never two separate "
            "networks on the representation that is confirmed to work."
        ),
    )
    parser.add_argument(
        "--daily-head", type=float, default=0.0,
        help=(
            "weight of an auxiliary loss on the THIRTY daily values of the "
            "target window. Every base in the pool is supervised by one scalar, "
            "the log of the 30-day sum. Predicting the daily profile gives the "
            "network thirty targets instead of one, so it is more supervision "
            "rather than a different view of the same supervision. The scored "
            "output stays the scalar head, so nothing about the metric changes."
        ),
    )
    parser.add_argument(
        "--calendar", action="store_true",
        help=(
            "give the network four calendar channels: sine and cosine of the "
            "day of year and of the day of week, for every day in the window. "
            "Without them the input is always 'the last 409 days' with no "
            "absolute date, so the network cannot know WHICH 30 days it is "
            "predicting and can only learn an averaged seasonal response. The "
            "boosting bases already receive the anchor's day of year through "
            "feats4; the networks, which carry most of the ensemble, never have."
        ),
    )
    parser.add_argument(
        "--representation", default="raw",
        choices=("raw", "diff", "occurrence", "daynorm", "cumulative", "rankday"),
        help=(
            "how the window is presented to the network. The shape "
            "representation is confirmed on public, so the question is which "
            "OTHER genuine changes of view exist."
        ),
    )
    parser.add_argument(
        "--per-user-scale", action="store_true",
        help=(
            "divide every user's window by that user's own mean activity, so "
            "the network sees SHAPE rather than level. Level is the dominant "
            "signal and every base in the pool is built around it; a shape-only "
            "model is a different representation of the same history."
        ),
    )
    parser.add_argument(
        "--valid-channel", action="store_true",
        help=(
            "append a channel marking real days against left padding. Without "
            "it an early anchor feeds the network a 409-day window that is "
            "mostly zeros, which is a different input distribution from the one "
            "at the competition anchor and measurably degrades the model."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("decay", "position", "position_decay", "event", "buyer"),
        default="decay",
    )
    return parser.parse_args()


class TCN(nn.Module):
    def __init__(
        self, channels: int, raw_channels: int, variant: str,
        length: int = 409, width: int = 96, daily_head: bool = False,
    ):
        super().__init__()
        self.length = length
        self.raw_channels = raw_channels
        self.variant = variant
        self.input = nn.Conv1d(channels, width, 5, padding=2)
        self.blocks = nn.ModuleList()
        for dilation in (1, 2, 4, 8, 16, 32, 64, 128):
            if dilation * 2 > length:
                break
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.BatchNorm1d(width),
            ))
        ages = torch.arange(length - 1, -1, -1, dtype=torch.float32)
        weights = torch.stack([torch.exp(-ages / decay) for decay in DECAYS])
        weights /= weights.sum(dim=1, keepdim=True)
        self.register_buffer("decay_weights", weights)
        pooled_width = width * (
            3 + (len(DECAYS) if variant in ("decay", "position_decay") else 0)
        )
        if variant == "event":
            pooled_width += raw_channels * 2
        self.head = nn.Sequential(
            nn.Linear(pooled_width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.daily_head = nn.Sequential(
            nn.Linear(pooled_width, 128), nn.GELU(), nn.Linear(128, 30),
        ) if daily_head else None
        self.buyer_head = (
            nn.Sequential(
                nn.Linear(pooled_width, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            if variant == "buyer" else None
        )

    def forward(self, values):
        hidden = torch.relu(self.input(values))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        pooled = [
            hidden.mean(-1),
            hidden.max(-1).values,
            hidden[..., -14:].mean(-1),
        ]
        if self.variant in ("decay", "position_decay"):
            pooled.extend([
                torch.sum(hidden * weights[None, None, :], dim=-1)
                for weights in self.decay_weights
            ])
        if self.variant == "event":
            raw = values[:, :self.raw_channels]
            positions = torch.linspace(
                0.0, 1.0, raw.shape[-1], device=raw.device, dtype=raw.dtype
            )
            event_positions = torch.where(
                raw > 0,
                positions[None, None, :],
                torch.full((), -1.0, device=raw.device, dtype=raw.dtype),
            )
            latest = torch.topk(event_positions, k=2, dim=-1).values
            recency = torch.where(latest[..., 0] >= 0, 1.0 - latest[..., 0], 1.0)
            gap = torch.where(
                latest[..., 1] >= 0,
                latest[..., 0] - latest[..., 1],
                1.0,
            )
            pooled.extend([recency, gap])
        pooled_values = torch.cat(pooled, dim=1)
        direct = self.head(pooled_values).squeeze(1)
        if self.daily_head is not None:
            return direct, self.daily_head(pooled_values)
        if self.buyer_head is not None:
            return direct, self.buyer_head(pooled_values).squeeze(1)
        return direct


def main():
    args = parse_args()
    root = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
    matrix_dir = Path(os.environ.get("ECUP_MAT", root / "work" / "mat"))
    output_dir = Path(os.environ.get("ECUP_OUT", root / "work"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    arrays = []
    for name in CHANNELS:
        values = np.load(matrix_dir / f"{name}.npy", mmap_mode="r")
        if args.nusers:
            values = values[:args.nusers]
        values = np.array(values, dtype=np.float32, copy=True)
        if name != "active":
            np.log1p(values, out=values)
        arrays.append(values.astype(np.float16))
    matrix = np.stack(arrays, axis=1)
    del arrays
    gc.collect()
    nusers, nchannels, ndays = matrix.shape
    if ndays != 409:
        raise ValueError(f"expected 409 days, received {ndays}")
    scale = np.float32(matrix[::37].astype(np.float32).reshape(-1, ndays).std() + 1e-6)
    gmv = np.load(matrix_dir / "gmv.npy", mmap_mode="r")[:nusers]
    cumulative = np.zeros((nusers, ndays + 1), dtype=np.float64)
    np.cumsum(gmv, axis=1, out=cumulative[:, 1:])
    del gmv

    validation_anchor = 378
    final_anchor = 408
    anchor_limit = validation_anchor - 29
    if args.anchor_end:
        anchor_limit = min(anchor_limit, args.anchor_end + 1)
    anchors = list(range(
        args.anchor_start + args.anchor_phase, anchor_limit, args.anchor_stride,
    ))
    if not anchors:
        raise ValueError("empty anchor set")
    # Bagging over users is the second untried diversity axis. The subset is
    # drawn once and held fixed for the run, so the model genuinely never sees
    # the held-out users rather than seeing them in a later epoch.
    training_users = np.arange(nusers)
    if args.user_fraction < 1.0:
        bag_rng = np.random.default_rng(args.user_seed or args.seed)
        training_users = np.sort(bag_rng.choice(
            nusers, int(nusers * args.user_fraction), replace=False
        ))
    targets = {
        anchor: np.log1p(
            cumulative[:, anchor + 31] - cumulative[:, anchor + 1]
        ).astype(np.float32)
        for anchor in anchors + [validation_anchor]
    }
    validation_target = targets[validation_anchor]
    daily_targets = {}
    if args.daily_head > 0:
        gmv_daily = np.load(matrix_dir / "gmv.npy", mmap_mode="r")[:nusers]
        for anchor in anchors:
            daily_targets[anchor] = np.log1p(
                np.asarray(gmv_daily[:, anchor + 1:anchor + 31], dtype=np.float32)
            )
        del gmv_daily
    split_rng = np.random.default_rng(20260825)
    calibration = np.sort(split_rng.choice(nusers, max(1, nusers // 5), replace=False))
    private_mask = np.ones(nusers, dtype=bool)
    private_mask[calibration] = False
    scoring = np.flatnonzero(private_mask)

    model = TCN(
        nchannels + (2 if args.variant in ("position", "position_decay") else 0)
        + (1 if args.valid_channel else 0) + (4 if args.calendar else 0),
        nchannels,
        args.variant,
        width=args.width,
        daily_head=args.daily_head > 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    batches = len(training_users) // args.batch_size
    planned_steps = args.epochs * len(anchors) * batches
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=max(1, total_steps), pct_start=0.15
    )

    # Population mean per channel per calendar day. Used only by the daynorm
    # representation, and computed from the inputs alone, never from a target.
    day_mean = matrix[::13].astype(np.float32).mean(axis=0)

    # Absolute calendar for every day the matrix covers, so a window ending at a
    # given anchor carries the true seasonal position of each of its days.
    epoch = date(2025, 1, 1)
    day_of_year = np.array(
        [(epoch + timedelta(days=i)).timetuple().tm_yday for i in range(ndays)],
        dtype=np.float32,
    )
    day_of_week = np.array(
        [(epoch + timedelta(days=i)).weekday() for i in range(ndays)],
        dtype=np.float32,
    )
    calendar_rows = np.stack([
        np.sin(2 * np.pi * day_of_year / 365.25),
        np.cos(2 * np.pi * day_of_year / 365.25),
        np.sin(2 * np.pi * day_of_week / 7.0),
        np.cos(2 * np.pi * day_of_week / 7.0),
    ]).astype(np.float32)

    def window(index, anchor):
        left = anchor - 408
        if left >= 0:
            output = matrix[index, :, left:anchor + 1].astype(np.float32)
        else:
            output = np.zeros((len(index), nchannels, 409), dtype=np.float32)
            output[:, :, -(anchor + 1):] = matrix[index, :, :anchor + 1].astype(np.float32)
        if args.representation == "diff":
            output = np.diff(output, axis=2, prepend=output[:, :, :1])
        elif args.representation == "occurrence":
            output = (output > 0).astype(np.float32)
        elif args.representation == "daynorm":
            output = output / (day_mean[None, :, :] + 1e-3)
        elif args.representation == "cumulative":
            output = np.cumsum(output, axis=2) / output.shape[2]
        elif args.representation == "rankday":
            order = np.argsort(np.argsort(output, axis=2), axis=2)
            output = order.astype(np.float32) / output.shape[2]
        if args.mask_recent:
            output[:, :, -args.mask_recent:] = 0.0
        if args.per_user_scale:
            # One scalar per user, taken over the whole window and all channels,
            # so relative channel magnitudes are preserved and only the user's
            # overall level is removed.
            level = output.reshape(len(index), -1).mean(axis=1)
            output = output / (level[:, None, None] + 1e-3)
        if args.representation in ("raw", "diff", "cumulative"):
            output /= scale
        if args.variant in ("position", "position_decay"):
            ages = np.arange(408, -1, -1, dtype=np.float32)
            position = np.stack([-ages / 408.0, np.exp(-ages / 30.0)])
            if left < 0:
                position[:, :-left] = 0.0
            position_batch = np.broadcast_to(
                position[None], (len(index), 2, 409)
            )
            output = np.concatenate([output, position_batch], axis=1)
        if args.calendar:
            block = np.zeros((4, 409), dtype=np.float32)
            if left >= 0:
                block[:] = calendar_rows[:, left:anchor + 1]
            else:
                block[:, -(anchor + 1):] = calendar_rows[:, :anchor + 1]
            output = np.concatenate(
                [output, np.broadcast_to(block[None], (len(index), 4, 409))], axis=1
            )
        if args.valid_channel:
            validity = np.ones((len(index), 1, 409), dtype=np.float32)
            if left < 0:
                validity[:, :, :-left] = 0.0
            output = np.concatenate([output, validity], axis=1)
        return output

    def predict(anchor):
        model.eval()
        prediction = np.empty(nusers, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, nusers, args.prediction_batch_size):
                index = np.arange(start, min(start + args.prediction_batch_size, nusers))
                values = torch.from_numpy(window(index, anchor)).to(device)
                output = model(values)
                if isinstance(output, tuple):
                    output = output[0]
                prediction[index] = output.float().cpu().numpy()
        model.train()
        if not np.isfinite(prediction).all():
            raise FloatingPointError("non-finite prediction")
        return prediction

    def calibrated_private_score(prediction):
        design = np.column_stack([prediction, np.ones(nusers)])
        coefficients = np.linalg.lstsq(
            design[calibration], validation_target[calibration], rcond=None
        )[0]
        calibrated = np.clip(design[scoring] @ coefficients, 0, None)
        score = float(np.sqrt(np.mean((validation_target[scoring] - calibrated) ** 2)))
        return score, coefficients

    config = {
        "tag": args.tag,
        "seed": args.seed,
        "architecture": f"exact_w409c_plus_{args.variant}",
        "only_change_from_w409c": args.variant,
        "channels": CHANNELS,
        "window": 409,
        "width": args.width,
        "anchors": anchors,
        "stride": args.anchor_stride,
        "anchor_phase": args.anchor_phase,
        "user_fraction": args.user_fraction,
        "user_seed": args.user_seed,
        "valid_channel": args.valid_channel,
        "per_user_scale": args.per_user_scale,
        "mask_recent": args.mask_recent,
        "representation": args.representation,
        "calendar": args.calendar,
        "daily_head": args.daily_head,
        "objective": args.objective,
        "head": "direct",
        "decays": DECAYS if args.variant in ("decay", "position_decay") else [],
        "buyer_auxiliary_loss_weight": 0.25 if args.variant == "buyer" else 0.0,
        "epochs": args.epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "private_selection": "fixed 50k calibration / independent 200k scoring",
    }
    (output_dir / f"{args.tag}_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)

    rng = np.random.default_rng(args.seed)
    history = []
    best_score = float("inf")
    started = time.time()
    step = 0
    stopped = False
    for epoch in range(1, args.epochs + 1):
        for anchor_value in rng.permutation(anchors):
            anchor = int(anchor_value)
            users = rng.permutation(training_users)
            for start in range(0, len(users) - args.batch_size + 1, args.batch_size):
                index = np.sort(users[start:start + args.batch_size])
                values = torch.from_numpy(window(index, anchor)).to(device)
                truth = torch.from_numpy(targets[anchor][index]).to(device)
                model_output = model(values)
                if args.daily_head > 0:
                    prediction, daily = model_output
                    daily_truth = torch.from_numpy(
                        daily_targets[anchor][index]
                    ).to(device)
                    loss = (
                        nn.functional.mse_loss(prediction, truth)
                        + args.daily_head * nn.functional.mse_loss(daily, daily_truth)
                    )
                elif isinstance(model_output, tuple):
                    prediction, buyer_logit = model_output
                    buyer = (truth > 0).to(truth.dtype)
                    loss = (
                        nn.functional.mse_loss(prediction, truth)
                        + 0.25 * nn.functional.binary_cross_entropy_with_logits(
                            buyer_logit, buyer
                        )
                    )
                else:
                    prediction = model_output
                    if args.objective == "probability":
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            prediction, (truth > 0).to(truth.dtype)
                        )
                    elif args.objective == "magnitude":
                        buyers = truth > 0
                        loss = (
                            nn.functional.mse_loss(prediction[buyers], truth[buyers])
                            if bool(buyers.any())
                            else prediction.sum() * 0.0
                        )
                    else:
                        loss = nn.functional.mse_loss(prediction, truth)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                step += 1
                if step == 1 or step % 300 == 0:
                    print(
                        f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.5f} "
                        f"elapsed={time.time()-started:.0f}s",
                        flush=True,
                    )
                if args.max_steps and step >= args.max_steps:
                    stopped = True
                    break
            if stopped:
                break
        validation_prediction = predict(validation_anchor)
        score, coefficients = calibrated_private_score(validation_prediction)
        history.append({
            "epoch": epoch,
            "private_score": score,
            "slope": float(coefficients[0]),
            "intercept": float(coefficients[1]),
        })
        print(f"EPOCH {epoch}: private_cal={score:.9f}", flush=True)
        if score < best_score:
            best_score = score
            np.save(output_dir / f"{args.tag}_val.npy", validation_prediction)
            np.save(output_dir / f"{args.tag}_final.npy", predict(final_anchor))
            torch.save(model.state_dict(), output_dir / f"{args.tag}_best.pt")
            print("  ^ saved new best", flush=True)
        (output_dir / f"{args.tag}_history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        if stopped:
            break
    print(f"DONE {args.tag}: best={best_score:.9f} elapsed={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
