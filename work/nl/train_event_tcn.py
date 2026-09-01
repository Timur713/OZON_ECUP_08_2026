#!/usr/bin/env python
"""H5 — event-indexed sequence base, the overnight GPU job.

Every network in the pool reads a DAY-indexed window: 409 slots, one per
calendar day, most of them empty for most users.  This one reads the last N
EVENT days, each carrying its own channel values plus the gap in days since the
previous event and its absolute age.  A user with 12 active days in 409 gets 12
informative slots instead of 409 mostly-zero ones.

That is the same restricted-view idea that worked for sw28 (+0.00011 real,
rejected only on the adaptive gate) applied along a different axis: not a
shorter window, but a different indexing of the same history.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[2]))
MAT = Path(os.environ.get("ECUP_MAT", ROOT / "work" / "mat"))
OUT = Path(os.environ.get("ECUP_OUT", ROOT / "work" / "cand"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVENTS = 64
HORIZON = 30
CHANNELS = ["gmv", "to_ord", "to_cart", "searches", "gmv_search", "gmv_cat",
            "search_to_ord", "cat_to_ord"]
SEED = int(os.environ.get("H5_SEED", "93"))
EPOCHS = 2
BATCH = 512
WIDTH = 128

torch.manual_seed(SEED)
np.random.seed(SEED)

raw = {name: np.load(MAT / f"{name}.npy", mmap_mode="r") for name in CHANNELS}
active = np.load(MAT / "active.npy", mmap_mode="r")
GMV = raw["gmv"]
USERS, DAYS = active.shape


def event_tensor(anchor, users_index):
    """Last EVENTS active days at or before `anchor`, newest last."""
    window = np.asarray(active[users_index, :anchor + 1], dtype=np.bool_)
    out = np.zeros((len(users_index), EVENTS, len(CHANNELS) + 2), dtype=np.float32)
    channel_values = {
        name: np.asarray(raw[name][users_index, :anchor + 1], dtype=np.float32)
        for name in CHANNELS
    }
    for row in range(len(users_index)):
        days = np.flatnonzero(window[row])
        if days.size == 0:
            continue
        days = days[-EVENTS:]
        slot = EVENTS - len(days)
        for c, name in enumerate(CHANNELS):
            values = channel_values[name][row, days]
            out[row, slot:, c] = np.log1p(np.maximum(values, 0.0))
        gaps = np.diff(days, prepend=days[0])
        out[row, slot:, len(CHANNELS)] = np.log1p(gaps)
        out[row, slot:, len(CHANNELS) + 1] = np.log1p(anchor - days)
    return out


class EventNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv1d(features, WIDTH, 3, padding=1), nn.GELU(),
            nn.Conv1d(WIDTH, WIDTH, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(WIDTH, WIDTH, 3, padding=4, dilation=4), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(WIDTH * 3, WIDTH), nn.GELU(), nn.Linear(WIDTH, 2)
        )

    def forward(self, x):
        h = self.encode(x.transpose(1, 2))
        pooled = torch.cat([h[:, :, -1], h.mean(-1), h.max(-1).values], dim=1)
        return self.head(pooled)


def target(anchor, users_index):
    window = GMV[users_index, anchor + 1:anchor + 1 + HORIZON]
    return np.log1p(window.sum(axis=1, dtype=np.float64)).astype(np.float32)


def run(train_anchors, predict_anchor, tag):
    model = EventNet(len(CHANNELS) + 2).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    order = np.arange(USERS)
    start = time.time()
    for epoch in range(EPOCHS):
        for anchor in train_anchors:
            np.random.shuffle(order)
            for i in range(0, USERS, BATCH * 8):
                chunk = np.sort(order[i:i + BATCH * 8])
                if chunk.size == 0:
                    continue
                x = torch.from_numpy(event_tensor(anchor, chunk)).to(DEVICE)
                y = torch.from_numpy(target(anchor, chunk)).to(DEVICE)
                for j in range(0, len(chunk), BATCH):
                    xb, yb = x[j:j + BATCH], y[j:j + BATCH]
                    out = model(xb)
                    direct = out[:, 0]
                    probability = torch.sigmoid(out[:, 1])
                    loss = (
                        ((direct - yb) ** 2).mean()
                        + 0.3 * nn.functional.binary_cross_entropy(
                            probability, (yb > 0).float()
                        )
                    )
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
            print(f"  {tag} epoch {epoch} anchor {anchor} "
                  f"loss {loss.item():.4f} [{time.time() - start:.0f}s]", flush=True)
    model.eval()
    prediction = np.zeros(USERS, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, USERS, BATCH * 8):
            chunk = np.arange(i, min(i + BATCH * 8, USERS))
            x = torch.from_numpy(event_tensor(predict_anchor, chunk)).to(DEVICE)
            prediction[chunk] = model(x)[:, 0].cpu().numpy()
    np.save(OUT / f"{tag}.npy", prediction)
    print(f"saved {tag} mean={prediction.mean():.4f} sd={prediction.std():.4f}",
          flush=True)
    return prediction


if __name__ == "__main__":
    for last, predict_anchor, suffix in ((342, 378, "val"), (378, 408, "final")):
        anchors = [t for t in range(186, last + 1, 24)]
        run(anchors, predict_anchor, f"h5_eventtcn_s{SEED}_{suffix}")
    print("DONE", flush=True)
