"""
Reference inference for the exported info-line model, using only NumPy + Pillow.

  python infer.py crop.png [--model ../public/models/info-line.json]

`crop.png` must be the card's info-line region already cut out and
straightened (see docs/info-line-model.md); any size is resized to the model's
input. Prints the decoded text, e.g. "SVI 001/198", and its confidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    dtype = "<f2" if manifest["dtype"] == "float16" else "<f4"
    weights = np.fromfile(manifest_path.parent / manifest["weights"], dtype=dtype).astype(np.float32)
    assert weights.size == manifest["params"], "weights file does not match manifest"
    return manifest, weights


def prepare(image: Image.Image, manifest) -> np.ndarray:
    """RGB image of the info-line region -> normalized (H, W) float32 input."""
    h, w = manifest["input"]["height"], manifest["input"]["width"]
    rgb = np.asarray(image.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    return np.clip((gray - lo) / max(hi - lo, 1.0), 0.0, 1.0) - 0.5


def conv2d(x, w, b, ph, pw):
    """x: (C, H, W); w: (O, C, kh, kw). Plain cross-correlation with zero padding."""
    c, h, wd = x.shape
    o, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    oh, ow = h + 2 * ph - kh + 1, wd + 2 * pw - kw + 1
    cols = np.empty((c, kh, kw, oh, ow), np.float32)
    for i in range(kh):
        for j in range(kw):
            cols[:, i, j] = xp[:, i : i + oh, j : j + ow]
    out = w.reshape(o, -1) @ cols.reshape(c * kh * kw, oh * ow)
    return out.reshape(o, oh, ow) + b[:, None, None]


def run(x: np.ndarray, manifest, weights):
    a = x[None]  # (1, H, W)
    for L in manifest["layers"]:
        if L["type"] == "maxpool":
            c, h, w = a.shape
            kh, kw = L["kh"], L["kw"]
            a = a[:, : h // kh * kh, : w // kw * kw].reshape(c, h // kh, kh, w // kw, kw).max(axis=(2, 4))
        else:
            n = L["outC"] * L["inC"] * L["kh"] * L["kw"]
            w = weights[L["w"] : L["w"] + n].reshape(L["outC"], L["inC"], L["kh"], L["kw"])
            b = weights[L["b"] : L["b"] + L["outC"]]
            a = conv2d(a, w, b, L["ph"], L["pw"])
            if L["relu"]:
                a = np.maximum(a, 0)
    logits = a[:, 0, :].T  # (T, classes)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    # Greedy CTC decode: argmax per step, merge repeats, drop blanks (class 0).
    best = probs.argmax(axis=1)
    text, confs, prev = "", [], 0
    for t, k in enumerate(best):
        if k != 0 and k != prev:
            text += manifest["charset"][k - 1]
            confs.append(probs[t, k])
        elif k != 0:
            confs[-1] = max(confs[-1], probs[t, k])
        prev = k
    return text, (min(confs) if confs else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent / "public" / "models" / "info-line.json"))
    args = ap.parse_args()
    manifest, weights = load(Path(args.model))
    text, conf = run(prepare(Image.open(args.image), manifest), manifest, weights)
    print(f"{text}\t{conf:.3f}")


if __name__ == "__main__":
    main()
