"""
Exports a trained checkpoint for the pure-TypeScript runtime:
  public/models/info-line.json  (architecture manifest)
  public/models/info-line.bin   (float16 weights, batch-norm folded in)

  python export.py runs/final/best.pt
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import CHARSET, INFO_LINE, INPUT_H, INPUT_W
from model import InfoLineNet

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    model = InfoLineNet(width=args.width)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    mods = list(model.body)
    layers = []
    blobs = []
    offset = 0

    def push(arr: np.ndarray) -> int:
        nonlocal offset
        start = offset
        flat = arr.astype(np.float16).ravel()
        blobs.append(flat)
        offset += flat.size
        return start

    i = 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach().numpy().astype(np.float64)
            b = m.bias.detach().numpy().astype(np.float64) if m.bias is not None else np.zeros(w.shape[0])
            relu = False
            j = i + 1
            if j < len(mods) and isinstance(mods[j], nn.BatchNorm2d):
                bn = mods[j]
                scale = bn.weight.detach().numpy() / np.sqrt(bn.running_var.detach().numpy() + bn.eps)
                w = w * scale[:, None, None, None]
                b = (b - bn.running_mean.detach().numpy()) * scale + bn.bias.detach().numpy()
                j += 1
            if j < len(mods) and isinstance(mods[j], nn.ReLU):
                relu = True
                j += 1
            layers.append({
                "type": "conv",
                "inC": int(w.shape[1]), "outC": int(w.shape[0]),
                "kh": int(w.shape[2]), "kw": int(w.shape[3]),
                "ph": int(m.padding[0]), "pw": int(m.padding[1]),
                "relu": relu,
                "w": push(w), "b": push(b),
            })
            i = j
        elif isinstance(m, nn.MaxPool2d):
            k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            layers.append({"type": "maxpool", "kh": int(k[0]), "kw": int(k[1])})
            i += 1
        elif isinstance(m, (nn.Dropout, nn.Dropout2d)):
            i += 1
        else:
            raise ValueError(f"unsupported layer {m}")

    # Sanity check: run the folded network in numpy against PyTorch.
    weights = np.concatenate(blobs).astype(np.float32)
    x = torch.randn(1, 1, INPUT_H, INPUT_W)
    with torch.no_grad():
        ref = model.body(x).numpy()[0, :, 0, :]
    a = x.numpy()[0]
    for L in layers:
        if L["type"] == "maxpool":
            C, H, W = a.shape
            a = a[:, : H // L["kh"] * L["kh"], : W // L["kw"] * L["kw"]].reshape(C, H // L["kh"], L["kh"], W // L["kw"], L["kw"]).max(axis=(2, 4))
        else:
            w = weights[L["w"]: L["w"] + L["outC"] * L["inC"] * L["kh"] * L["kw"]].reshape(L["outC"], L["inC"], L["kh"], L["kw"])
            b = weights[L["b"]: L["b"] + L["outC"]]
            out = torch.nn.functional.conv2d(torch.from_numpy(a)[None], torch.from_numpy(w), torch.from_numpy(b), padding=(L["ph"], L["pw"]))[0].numpy()
            a = np.maximum(out, 0) if L["relu"] else out
    err = np.abs(a[:, 0, :] - ref).max()
    print(f"folded network max abs diff vs PyTorch: {err:.4f} (float16 weights)")

    out_dir = ROOT / "public" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.concatenate(blobs).astype("<f2").tofile(out_dir / "info-line.bin")
    manifest = {
        "name": "info-line",
        "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": args.note,
        "input": {"height": INPUT_H, "width": INPUT_W, "region": dict(zip("xywh", INFO_LINE)), "normalization": "gray-p2-p98-centered"},
        "charset": CHARSET,
        "dtype": "float16",
        "weights": "info-line.bin",
        "params": int(offset),
        "layers": layers,
    }
    (out_dir / "info-line.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out_dir}/info-line.json and info-line.bin ({offset * 2 / 1024:.0f} KiB, {len(layers)} layers)")


if __name__ == "__main__":
    main()
