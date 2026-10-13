"""
On-the-fly synthetic training data for the info-line model.

Each sample takes an official card scan, simulates how its bottom-left info
line looks through a phone/laptop camera (resolution, blur, noise, JPEG,
lighting, glare), then rectifies it with a deliberately imperfect quad
(simulating card-edge detection error) exactly the way the app does.
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from codebox import find_code_box, find_number_text, random_code, random_number, render_code_box, replace_number
from common import CARD_ASPECT, INFO_LINE, INPUT_H, INPUT_W, encode

# Margin (card widths) kept around the info line when pre-cropping scans so
# jittered quads stay inside real pixels.
MARGIN = 0.06


def load_scan_crops(scan_dir: Path, codes_include=None, codes_exclude=None):
    """Loads every scan once and keeps only the area around the info line."""
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    items = []
    for entry in manifest:
        if codes_include is not None and entry["code"] not in codes_include:
            continue
        if codes_exclude is not None and entry["code"] in codes_exclude:
            continue
        im = cv2.imread(str(scan_dir / entry["file"]), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        H, W = im.shape[:2]
        x, y, w, h = INFO_LINE
        x0 = max(0, int((x - MARGIN) * W))
        y0 = max(0, int((y - MARGIN * CARD_ASPECT) * H))
        x1 = min(W, int((x + w + MARGIN) * W))
        y1 = H
        crop = im[y0:y1, x0:x1]
        if crop.shape[2] == 4:
            alpha = crop[:, :, 3]
            rgb = crop[:, :, :3]
        else:
            alpha = np.full(crop.shape[:2], 255, np.uint8)
            rgb = crop
        box = find_code_box(im[:, :, :3])
        num = find_number_text(np.ascontiguousarray(im[:, :, :3]), box)
        if num is not None:
            num = {**num, "x": num["x"] - x0, "y": num["y"] - y0, "mask_xy": (num["mask_xy"][0] - x0, num["mask_xy"][1] - y0)}
        items.append({
            "num": num,
            # Set-code box relative to the crop, for synthetic code replacement.
            "box": (box[0] - x0, box[1] - y0, box[2], box[3]) if box else None,
            "rgb": np.ascontiguousarray(rgb),
            "alpha": alpha,
            "offset": (x0, y0),
            "size": (W, H),
            "label": entry["label"],
            "code": entry["code"],
        })
    return items


def _rand_bg(shape, rng):
    base = rng.integers(0, 256, 3)
    bg = np.empty(shape, np.uint8)
    bg[:] = base
    return bg


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    """Percentile contrast stretch to [0,1], then centre. Mirrors line-model.ts."""
    g = gray.astype(np.float32)
    lo = np.percentile(g, 2)
    hi = np.percentile(g, 98)
    g = np.clip((g - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return g - 0.5


def to_gray(rgb_bgr: np.ndarray) -> np.ndarray:
    b, g, r = rgb_bgr[:, :, 0].astype(np.float32), rgb_bgr[:, :, 1].astype(np.float32), rgb_bgr[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


class InfoLineDataset(Dataset):
    def __init__(
        self,
        items,
        length: int,
        seed: int = 0,
        difficulty: float | None = None,
        clean: bool = False,
        synth_code: float = 0.0,
        synth_number: float = 0.0,
        negative: float = 0.0,
    ):
        """
        difficulty: fixed 0..1.3 degradation strength, or None to sample one per example.
        synth_code: probability of replacing the set-code box with a random synthetic code.
        synth_number: probability of replacing the collector number with a random one.
        negative: probability of a misplaced / upside-down crop labelled "" (nothing to read).
        """
        self.synth_code = synth_code
        self.synth_number = synth_number
        self.negative = negative
        self.items = items
        self.length = length
        self.seed = seed
        self.difficulty = difficulty
        self.clean = clean

    def __len__(self):
        return self.length

    def render(self, item, rng: np.random.Generator):
        # Per-sample difficulty (mostly moderate, some very easy / very hard).
        k = self.difficulty if self.difficulty is not None else float(rng.beta(2.0, 2.0) * 1.3)
        W, H = item["size"]
        ox, oy = item["offset"]

        if self.clean:
            cam_w = 900.0
        else:
            # Card width in camera pixels (log-uniform), down to 380px for the hardest samples.
            lo = 900 - 520 * min(k, 1.0)
            cam_w = math.exp(rng.uniform(math.log(lo), math.log(1300)))
        s = cam_w / W

        # Composite the transparent rounded corner onto a random background.
        rgb = item["rgb"]
        alpha = item["alpha"].astype(np.float32)[:, :, None] / 255.0
        bg = _rand_bg(rgb.shape, rng)
        img = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

        label = item["label"]
        if item["box"] is not None and rng.random() < self.synth_code:
            bx, by, bw, bh = item["box"]
            code = random_code(rng)
            box_bgr, box_a = render_code_box(bw, bh, code, rng)
            a = box_a.astype(np.float32)[:, :, None] / 255.0
            region = img[by : by + bh, bx : bx + bw]
            img[by : by + bh, bx : bx + bw] = (box_bgr * a + region * (1 - a)).astype(np.uint8)
            label = code + label[3:]
        if item["num"] is not None and rng.random() < self.synth_number:
            new = random_number(rng, promo=rng.random() < 0.1)
            replace_number(img, item["num"], label[4:], new, rng)
            label = label[:4] + new
        negative = None
        if not self.clean and rng.random() < self.negative:
            negative = "flip" if rng.random() < 0.5 else "shift"
            label = ""

        # Into "camera space".
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)

        if not self.clean:
            f = img.astype(np.float32)
            # Lighting: exposure, per-channel white balance, gamma, gradient.
            exposure = rng.uniform(1 - 0.45 * k, 1 + 0.25 * k)
            wb = rng.uniform(1 - 0.12 * k, 1 + 0.12 * k, 3)
            f *= exposure * wb
            f = np.clip(f, 0, 255)
            gamma = rng.uniform(1 - 0.3 * k, 1 + 0.4 * k)
            f = 255.0 * (f / 255.0) ** gamma
            hh, ww = f.shape[:2]
            ang = rng.uniform(0, 2 * math.pi)
            yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
            grad = 1 + rng.uniform(0, 0.35 * k) * (((xx - ww / 2) * math.cos(ang) + (yy - hh / 2) * math.sin(ang)) / max(hh, ww))
            f *= grad[:, :, None]
            # Glare / holo sheen.
            if rng.random() < 0.3 * k:
                cx, cy = rng.uniform(0, ww), rng.uniform(0, hh)
                rad = rng.uniform(0.05, 0.4) * ww
                strength = rng.uniform(40, 200)
                blob = strength * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (rad * rad)))
                tint = rng.uniform(0.8, 1.0, 3)
                f += blob[:, :, None] * tint
            img = np.clip(f, 0, 255).astype(np.uint8)
            # Optics: defocus and occasional motion blur.
            sigma = rng.uniform(0.2, max(0.3, 1.6 * k))
            if sigma > 0.3:
                img = cv2.GaussianBlur(img, (0, 0), sigma)
            if rng.random() < 0.25 * k:
                ln = int(rng.integers(2, 6))
                kern = np.zeros((ln, ln), np.float32)
                kern[ln // 2, :] = 1.0 / ln
                rot = cv2.getRotationMatrix2D((ln / 2 - 0.5, ln / 2 - 0.5), rng.uniform(0, 180), 1)
                kern = cv2.warpAffine(kern, rot, (ln, ln))
                kern /= max(kern.sum(), 1e-6)
                img = cv2.filter2D(img, -1, kern)
            # Sensor noise + compression.
            noise = rng.normal(0, rng.uniform(0, 9 * k), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            img = _jpeg(img, rng.uniform(45, 95))

        # Ideal region corners (card coords) -> camera pixels.
        x, y, w, h = INFO_LINE
        card_h = 1.0  # normalized
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64)
        if not self.clean:
            # Detection error, matched to the measured corner error of
            # detect.ts (median ~0.2% of card width) with occasional outliers.
            # Units: card width (horizontal) and card height (vertical).
            tail = 2.5 if rng.random() < 0.1 else 1.0
            dx = rng.normal(0, 0.004 * tail)
            dy = rng.normal(0, 0.0025 * tail)
            sc = rng.normal(1.0, 0.015 * tail)
            rot = math.radians(rng.normal(0, 0.6 * tail))
            c = corners.mean(axis=0)
            rel = corners - c
            # Rotate in physical units (card height = 1/aspect widths).
            rel_phys = rel * np.array([1.0, 1.0 / CARD_ASPECT])
            rr = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
            rel_phys = rel_phys @ rr.T * sc
            rel = rel_phys / np.array([1.0, 1.0 / CARD_ASPECT])
            corners = c + rel + np.array([dx, dy * card_h])
            corners += rng.normal(0, 0.0008 * tail, corners.shape)
            if negative == "shift":
                # Text mostly out of the crop: there is nothing to read.
                corners[:, 1] += rng.choice([-1, 1]) * rng.uniform(0.022, 0.04)
        src = np.stack([corners[:, 0] * W * s - ox * s, corners[:, 1] * H * s - oy * s], axis=1).astype(np.float32)
        dst = np.array([[0, 0], [INPUT_W * 2, 0], [INPUT_W * 2, INPUT_H * 2], [0, INPUT_H * 2]], np.float32)
        M = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(
            img, M, (INPUT_W * 2, INPUT_H * 2), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )
        warped = cv2.resize(warped, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        if negative == "flip":
            # Upside-down info line (what the "flipped" hypothesis sees on an upright card).
            warped = np.ascontiguousarray(warped[::-1, ::-1])
        return normalize_gray(to_gray(warped)), label

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, idx, torch.initial_seed() % (2**31) if not self.clean else 0))
        item = self.items[int(rng.integers(len(self.items)))] if not self.clean else self.items[idx % len(self.items)]
        x, label = self.render(item, rng)
        return torch.from_numpy(x)[None], label


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    targets = torch.tensor([i for l in labels for i in encode(l)], dtype=torch.long)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return xs, targets, lengths, labels
