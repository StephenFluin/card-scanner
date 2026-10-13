"""Breaks a checkpoint's errors on a rendered set down by field: python analyze.py runs/x/last.pt val-holdout [--width 1.5]"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from common import decode_greedy
from model import InfoLineNet

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("dataset")
ap.add_argument("--width", type=float, default=1.0)
args = ap.parse_args()

m = InfoLineNet(width=args.width).cuda()
m.load_state_dict(torch.load(args.checkpoint))
m.eval()
imgs = np.load(ROOT / ".cache" / "dataset" / args.dataset / "images.npy")
labels = json.loads((ROOT / ".cache" / "dataset" / args.dataset / "labels.json").read_text())
preds = []
with torch.no_grad():
    for i in range(0, len(labels), 2048):
        x = torch.from_numpy(imgs[i : i + 2048]).cuda().float().unsqueeze(1) / 255 - 0.5
        logp = m(x)
        probs = logp.exp().max(-1).values.T.cpu().numpy()
        for row, pr in zip(logp.argmax(-1).T.cpu().numpy(), probs):
            preds.append((decode_greedy(row), float(pr[row != 0].min()) if (row != 0).any() else 0.0))
n = len(labels)
exact = sum(p == l for (p, _), l in zip(preds, labels))
code = sum(p[:3] == l[:3] for (p, _), l in zip(preds, labels))
num = sum(p[4:] == l[4:] for (p, _), l in zip(preds, labels))
print(f"{args.dataset}: exact {exact / n:.3f}  code {code / n:.3f}  number {num / n:.3f}")
# Confidence-thresholded precision: how often a confident read is right.
for t in (0.5, 0.7, 0.9):
    conf = [(p == l) for (p, c), l in zip(preds, labels) if c >= t and p]
    print(f"  conf>={t}: coverage {len(conf) / n:.3f}  precision {np.mean(conf) if conf else 0:.4f}")
print("  wrong codes:", collections.Counter(p[:3] for (p, _), l in zip(preds, labels) if p[:3] != l[:3]).most_common(10))
print("  sample number errors:", [f"{l}->{p}" for (p, _), l in zip(preds, labels) if p[4:] != l[4:]][:10])
