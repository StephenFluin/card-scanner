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
        items.append({
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
    def __init__(self, items, length: int, seed: int = 0, difficulty: float = 1.0, clean: bool = False):
        self.items = items
        self.length = length
        self.seed = seed
        self.difficulty = difficulty
        self.clean = clean

    def __len__(self):
        return self.length

    def render(self, item, rng: np.random.Generator):
        k = self.difficulty
        W, H = item["size"]
        ox, oy = item["offset"]

        if self.clean:
            cam_w = 900.0
        else:
            # Card width in camera pixels (log-uniform): 380px (far / low-res) .. 1300px.
            cam_w = math.exp(rng.uniform(math.log(380), math.log(1300)))
        s = cam_w / W

        # Composite the transparent rounded corner onto a random background.
        rgb = item["rgb"]
        alpha = item["alpha"].astype(np.float32)[:, :, None] / 255.0
        bg = _rand_bg(rgb.shape, rng)
        img = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

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
            sigma = rng.uniform(0.2, 1.6 * k)
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
            # Detection error: translate / scale / rotate the region a little,
            # in units of card width (horizontal) and card height (vertical).
            dx = rng.normal(0, 0.007 * k)
            dy = rng.normal(0, 0.006 * k)
            sc = rng.normal(1.0, 0.035 * k)
            rot = math.radians(rng.normal(0, 1.2 * k))
            c = corners.mean(axis=0)
            rel = corners - c
            # Rotate in physical units (card height = 1/aspect widths).
            rel_phys = rel * np.array([1.0, 1.0 / CARD_ASPECT])
            rr = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
            rel_phys = rel_phys @ rr.T * sc
            rel = rel_phys / np.array([1.0, 1.0 / CARD_ASPECT])
            corners = c + rel + np.array([dx, dy * card_h])
            corners += rng.normal(0, 0.0015 * k, corners.shape)
        src = np.stack([corners[:, 0] * W * s - ox * s, corners[:, 1] * H * s - oy * s], axis=1).astype(np.float32)
        dst = np.array([[0, 0], [INPUT_W * 2, 0], [INPUT_W * 2, INPUT_H * 2], [0, INPUT_H * 2]], np.float32)
        M = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(
            img, M, (INPUT_W * 2, INPUT_H * 2), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )
        warped = cv2.resize(warped, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        return normalize_gray(to_gray(warped))

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, idx, torch.initial_seed() % (2**31) if not self.clean else 0))
        item = self.items[int(rng.integers(len(self.items)))] if not self.clean else self.items[idx % len(self.items)]
        x = self.render(item, rng)
        return torch.from_numpy(x)[None], item["label"]


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    targets = torch.tensor([i for l in labels for i in encode(l)], dtype=torch.long)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return xs, targets, lengths, labels
