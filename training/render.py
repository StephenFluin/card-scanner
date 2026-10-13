"""
Pre-renders augmented info-line samples to disk so training is GPU-bound
instead of CPU-bound (on-the-fly augmentation manages ~1.4k samples/s on 8 cores).

  python render.py --name train-holdout --per-card 60 --exclude PBL,DRI
  python render.py --name val-holdout   --per-card 20 --include PBL,DRI --seed 7
  python render.py --name train-all     --per-card 60

Writes .cache/dataset/<name>/images.npy (uint8, N x H x W, already
normalized: 0..255 maps to -0.5..0.5) and labels.json.
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from common import INPUT_H, INPUT_W
from codebox import GlyphBank
from dataset import InfoLineDataset, load_scan_crops

ROOT = Path(__file__).resolve().parent.parent
_items = None
_args = None


def _init(items, args):
    global _items, _args
    import cv2

    cv2.setNumThreads(1)
    _items = items
    _args = args


def _render(job):
    idx, seed = job
    item = _items[idx % len(_items)]
    ds = InfoLineDataset([item], 1, seed=seed, difficulty=_args["difficulty"], clean=_args["clean"], synth_code=_args["synth_code"], synth_number=_args["synth_number"], negative=_args["negative"], bank=_args["bank"])
    rng = np.random.default_rng((seed, idx))
    x, label = ds.render(item, rng)
    return np.clip(np.round((x + 0.5) * 255), 0, 255).astype(np.uint8), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-card", type=int, default=60)
    ap.add_argument("--include", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--difficulty", type=float, default=None)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--synth-code", type=float, default=0.0, help="probability of a random synthetic set code")
    ap.add_argument("--synth-number", type=float, default=0.0, help="probability of a random synthetic collector number")
    ap.add_argument("--negative", type=float, default=0.0, help="probability of a misplaced crop labelled empty")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    include = set(args.include.split(",")) if args.include else None
    exclude = set(args.exclude.split(",")) if args.exclude else None
    bank = GlyphBank()
    items = load_scan_crops(ROOT / ".cache" / "scans", codes_include=include, codes_exclude=exclude, bank=bank)
    print(f"real glyph bank: {bank.chars} ({sum(len(v) for v in bank.by_char.values())} glyphs)")
    n = len(items) * args.per_card
    print(f"{len(items)} scans x {args.per_card} = {n} samples at {INPUT_H}x{INPUT_W}")
    out = ROOT / ".cache" / "dataset" / args.name
    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(n, INPUT_H, INPUT_W))
    labels = [""] * n
    jobs = [(i, args.seed) for i in range(n)]
    t0 = time.time()
    shared = {"difficulty": args.difficulty, "clean": args.clean, "synth_code": args.synth_code, "synth_number": args.synth_number, "negative": args.negative, "bank": bank}
    # 'spawn': forking after PIL/FreeType or OpenCV initialized can deadlock workers.
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(items, shared)) as pool:
        for i, (img, label) in enumerate(pool.imap(_render, jobs, chunksize=256)):
            images[i] = img
            labels[i] = label
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{n} ({(i + 1) / (time.time() - t0):.0f}/s)", flush=True)
    images.flush()
    (out / "labels.json").write_text(json.dumps(labels))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
