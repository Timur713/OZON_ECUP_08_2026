#!/usr/bin/env python
"""Exact-backbone scaling study: window length, width and anchor density.

The structural round on this backbone is closed: decay, position, buyer and
combined position+decay all turned out to re-express the same representation.
FINDINGS 2.3 says the axes that actually create diversity are WINDOW and SEED,
and 2.6 measured "two epochs then overfit" on a sparse 14-anchor schedule, so
anchor density has never been separated from epoch count.  This trainer opens
exactly those knobs and adds a `plain` variant with no structural addition.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


CHANNELS = ["gmv", "to_ord", "to_cart", "searches", "active", "search", "cat"]
CHANNELS_ALL = CHANNELS + [
    "gmv_cat", "gmv_search",
    "has_search_to_ord", "has_search_to_cart",
    "has_cat_to_ord", "has_cat_to_cart",
    "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
]
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
    parser.add_argument(
        "--variant",
        choices=("plain", "decay", "position", "position_decay", "event", "buyer"),
        default="plain",
    )
    parser.add_argument("--window", type=int, default=409)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--anchor-start", type=int, default=186)
    parser.add_argument(
        "--anchor-end", type=int, default=0,
        help="last training anchor; 0 = validation_anchor-30 (never sees the fold)",
    )
    parser.add_argument("--anchor-stride", type=int, default=12)
    parser.add_argument("--channels", choices=("base", "all"), default="base")
    parser.add_argument(
        "--channel-list", default="",
        help="explicit comma-separated channel names; overrides --channels. "
             "A restricted view makes the model WEAK but structurally different, "
             "which is what the ridge actually pays for (FINDINGS 2.4).",
    )
    return parser.parse_args()


class TCN(nn.Module):
    def __init__(
        self, channels: int, raw_channels: int, variant: str,
        length: int = 409, width: int = 96,
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
    if args.channel_list:
        channel_names = [name.strip() for name in args.channel_list.split(",") if name.strip()]
        unknown = [n for n in channel_names if n not in CHANNELS_ALL]
        if unknown:
            raise ValueError(f"unknown channels: {unknown}")
    else:
        channel_names = CHANNELS_ALL if args.channels == "all" else CHANNELS
    for name in channel_names:
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
    anchor_end = args.anchor_end or (validation_anchor - 30)
    if anchor_end + 30 > final_anchor:
        raise ValueError("training anchor target window must be fully observed")
    anchors = list(range(args.anchor_start, anchor_end + 1, args.anchor_stride))
    if args.anchor_end and anchor_end + 30 > validation_anchor:
        print(
            f"NOTE anchor_end={anchor_end} overlaps the validation fold; "
            "fold378 score is NOT a clean holdout for this run",
            flush=True,
        )
    targets = {
        anchor: np.log1p(
            cumulative[:, anchor + 31] - cumulative[:, anchor + 1]
        ).astype(np.float32)
        for anchor in anchors + [validation_anchor]
    }
    validation_target = targets[validation_anchor]
    split_rng = np.random.default_rng(20260825)
    calibration = np.sort(split_rng.choice(nusers, max(1, nusers // 5), replace=False))
    private_mask = np.ones(nusers, dtype=bool)
    private_mask[calibration] = False
    scoring = np.flatnonzero(private_mask)

    model = TCN(
        nchannels + (2 if args.variant in ("position", "position_decay") else 0),
        nchannels,
        args.variant,
        length=args.window,
        width=args.width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    batches = nusers // args.batch_size
    planned_steps = args.epochs * len(anchors) * batches
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=max(1, total_steps), pct_start=0.15
    )

    span = args.window

    def window(index, anchor):
        left = anchor - (span - 1)
        if left >= 0:
            output = matrix[index, :, left:anchor + 1].astype(np.float32)
        else:
            output = np.zeros((len(index), nchannels, span), dtype=np.float32)
            output[:, :, -(anchor + 1):] = matrix[index, :, :anchor + 1].astype(np.float32)
        output /= scale
        if args.variant in ("position", "position_decay"):
            ages = np.arange(span - 1, -1, -1, dtype=np.float32)
            position = np.stack([-ages / (span - 1.0), np.exp(-ages / 30.0)])
            if left < 0:
                position[:, :-left] = 0.0
            position_batch = np.broadcast_to(
                position[None], (len(index), 2, span)
            )
            output = np.concatenate([output, position_batch], axis=1)
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
        "architecture": f"exact_backbone_{args.variant}_w{args.window}",
        "only_change_from_w409c": args.variant,
        "channels": channel_names,
        "window": args.window,
        "width": args.width,
        "anchors": anchors,
        "anchor_stride": args.anchor_stride,
        "anchor_count": len(anchors),
        "stride": args.anchor_stride,
        "user_fraction": 1.0,
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
            users = rng.permutation(nusers)
            for start in range(0, nusers - args.batch_size + 1, args.batch_size):
                index = np.sort(users[start:start + args.batch_size])
                values = torch.from_numpy(window(index, anchor)).to(device)
                truth = torch.from_numpy(targets[anchor][index]).to(device)
                model_output = model(values)
                if isinstance(model_output, tuple):
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
