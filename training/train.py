"""
Trains the info-line model on pre-rendered datasets (see render.py), entirely
from GPU memory.

  python train.py --train train-holdout --val val-holdout,val-seen-clean --name holdout
  python train.py --train train-all --val val-seen-clean --name final
  python train.py --train train-holdout --overfit 256      # sanity check: must reach ~100%

Checkpoints go to training/runs/<name>/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import decode_greedy, encode
from model import InfoLineNet

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / ".cache" / "dataset"


def load(name, device):
    images = torch.from_numpy(np.load(DATA / name / "images.npy", mmap_mode="r")[:]).to(device)
    labels = json.loads((DATA / name / "labels.json").read_text())
    enc = [encode(l) for l in labels]
    max_len = max(len(e) for e in enc)
    targets = torch.zeros((len(enc), max_len), dtype=torch.long)
    for i, e in enumerate(enc):
        targets[i, : len(e)] = torch.tensor(e)
    lengths = torch.tensor([len(e) for e in enc], dtype=torch.long)
    return images, targets.to(device), lengths.to(device), labels


def to_input(images_u8):
    return images_u8.float().unsqueeze(1) / 255.0 - 0.5


def gpu_augment(x, strength=1.0):
    """Small random affine on top of the pre-rendered augmentation, for extra variety."""
    n = x.shape[0]
    dev = x.device
    s = 1 + (torch.rand(n, device=dev) - 0.5) * 0.06 * strength
    tx = (torch.rand(n, device=dev) - 0.5) * 0.04 * strength
    ty = (torch.rand(n, device=dev) - 0.5) * 0.08 * strength
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    # Random contrast / inversion-free brightness wiggle, re-centred.
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3 * strength
    return (x * c).clamp(-0.5, 0.5)


@torch.no_grad()
def evaluate(model, images, labels, batch=2048, show=0):
    model.eval()
    correct = 0
    errors = []
    for i in range(0, len(labels), batch):
        x = to_input(images[i : i + batch])
        pred = model(x).argmax(-1).T.cpu().numpy()
        for p, l in zip(pred, labels[i : i + batch]):
            t = decode_greedy(p)
            if t == l:
                correct += 1
            elif len(errors) < show:
                errors.append(f"{l!r} -> {t!r}")
    model.train()
    return correct / max(1, len(labels)), errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--overfit", type=int, default=0, help="train on only the first N samples (sanity check)")
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cuda"
    name = args.name or args.train
    out = ROOT / "training" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    images, targets, lengths, labels = load(args.train, device)
    # Hold back a slice of the training scans' renders to measure seen-set accuracy.
    perm = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0)).to(device)
    if args.overfit:
        train_idx = perm[: args.overfit]
        seen_idx = train_idx
    else:
        seen_idx = perm[:4000]
        train_idx = perm[4000:]
    vals = {}
    for v in filter(None, args.val.split(",")):
        vi, _, _, vl = load(v, device)
        vals[v] = (vi, vl)
    seen_labels = [labels[i] for i in seen_idx.tolist()]
    print(f"train {len(train_idx)} samples, val {', '.join(f'{k}={len(v[1])}' for k, v in vals.items())}; loaded in {time.time() - t0:.0f}s")

    model = InfoLineNet(width=args.width).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    best = -1.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = train_idx[torch.randint(len(train_idx), (args.batch,), device=device)]
        x = to_input(images[idx])
        if not args.overfit:
            x = gpu_augment(x)
        logp = model(x)
        tl = lengths[idx]
        tg = targets[idx]
        flat = tg[torch.arange(tg.shape[1], device=device)[None, :] < tl[:, None]]
        loss = ctc(logp, flat, torch.full((x.shape[0],), logp.shape[0], dtype=torch.long, device=device), tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(f"step {step:6d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step * 1000:.1f} ms/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            acc_seen, _ = evaluate(model, images[seen_idx], seen_labels)
            msg = f"== step {step}: seen {acc_seen:.4f}"
            score = acc_seen
            shown = []
            for k, (vi, vl) in vals.items():
                acc, errs = evaluate(model, vi, vl, show=6)
                msg += f"  {k} {acc:.4f}"
                shown += errs
                if "holdout" in k:
                    score = acc
            print(msg, flush=True)
            for e in shown:
                print("    ", e)
            torch.save(model.state_dict(), out / "last.pt")
            if score > best:
                best = score
                torch.save(model.state_dict(), out / "best.pt")
    print(f"done in {(time.time() - t0) / 60:.1f} min, best {best:.4f}; checkpoints in {out}")


if __name__ == "__main__":
    main()
