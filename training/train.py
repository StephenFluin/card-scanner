"""
Trains the info-line model.

  python train.py --holdout PBL,DRI      # measure accuracy on sets never seen in training
  python train.py --holdout none          # final model on everything

Writes checkpoints to training/runs/<name>/.
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import decode_greedy
from dataset import InfoLineDataset, collate, load_scan_crops
from model import InfoLineNet

ROOT = Path(__file__).resolve().parent.parent


def worker_init(_):
    # One thread per data worker; otherwise OpenCV/BLAS oversubscribe the CPU.
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    errors = []
    with torch.no_grad():
        for b, (x, _, _, labels) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            logp = model(x.to(device))
            pred = logp.argmax(-1).T.cpu().numpy()
            for p, l in zip(pred, labels):
                text = decode_greedy(p)
                total += 1
                if text == l:
                    correct += 1
                elif len(errors) < 12:
                    errors.append(f"{l!r} -> {text!r}")
    model.train()
    return correct / max(total, 1), errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="PBL,DRI", help="comma separated set codes, or 'none'")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    holdout = set() if args.holdout == "none" else set(args.holdout.split(","))
    name = args.name or ("final" if not holdout else "holdout-" + "-".join(sorted(holdout)))
    out = ROOT / "training" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scans = ROOT / ".cache" / "scans"
    t0 = time.time()
    train_items = load_scan_crops(scans, codes_exclude=holdout)
    val_items = load_scan_crops(scans, codes_include=holdout) if holdout else None
    print(f"loaded {len(train_items)} train scans" + (f", {len(val_items)} holdout ({','.join(sorted(holdout))})" if holdout else "") + f" in {time.time() - t0:.0f}s")

    train_ds = InfoLineDataset(train_items, length=args.steps * args.batch, seed=1)
    loader = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate, persistent_workers=True, prefetch_factor=4, worker_init_fn=worker_init)
    # Fixed-seed validation sets: seen sets with augmentation, and holdout sets.
    val_seen = DataLoader(InfoLineDataset(train_items, length=2048, seed=999), batch_size=512, num_workers=4, collate_fn=collate, worker_init_fn=worker_init)
    val_hold = DataLoader(InfoLineDataset(val_items, length=2048, seed=999), batch_size=512, num_workers=4, collate_fn=collate, worker_init_fn=worker_init) if val_items else None
    val_easy = DataLoader(InfoLineDataset(val_items or train_items, length=1024, seed=7, difficulty=0.4), batch_size=512, num_workers=4, collate_fn=collate, worker_init_fn=worker_init)

    model = InfoLineNet(width=args.width).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    step = 0
    best = -1.0
    t0 = time.time()
    for x, targets, lengths, _ in loader:
        x = x.to(device, non_blocking=True)
        logp = model(x)
        input_lengths = torch.full((x.shape[0],), logp.shape[0], dtype=torch.long)
        loss = ctc(logp, targets, input_lengths, lengths)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        if step % 250 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / step * 1000:.0f} ms/step", flush=True)
        if step % 2500 == 0 or step == args.steps:
            acc_seen, _ = evaluate(model, val_seen, device)
            msg = f"== step {step}: seen-sets acc {acc_seen:.3f}"
            score = acc_seen
            if val_hold:
                acc_hold, errs = evaluate(model, val_hold, device)
                acc_easy, _ = evaluate(model, val_easy, device)
                msg += f"  holdout acc {acc_hold:.3f}  holdout-easy acc {acc_easy:.3f}"
                score = acc_hold
                print(msg)
                for e in errs:
                    print("   ", e)
            else:
                print(msg)
            torch.save(model.state_dict(), out / "last.pt")
            if score > best:
                best = score
                torch.save(model.state_dict(), out / "best.pt")
        if step >= args.steps:
            break
    print(f"done in {(time.time() - t0) / 60:.1f} min, best {best:.3f}; checkpoints in {out}")


if __name__ == "__main__":
    main()
