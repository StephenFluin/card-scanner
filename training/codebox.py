"""
Synthetic set-code boxes.

There are only ~25 distinct set codes in the training scans, so a model trained
on them alone memorizes the codes as whole words and cannot read a new set's
code. We paste look-alike boxes ("XYZ EN": white condensed caps on a black
rounded box with a thin light outline) with random codes over the real box,
so the model has to learn individual letters.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Extra open-licensed (OFL) condensed fonts from Google Fonts, downloaded into
# .cache/fonts (training data only; see training/README.md).
EXTRA_FONTS = Path(__file__).resolve().parent.parent / ".cache" / "fonts"


def _existing(paths):
    return [str(p) for p in paths if Path(p).exists()]


FONTS = _existing(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/ibm-plex/IBMPlexSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    ]
    + [EXTRA_FONTS / f for f in (
        "BarlowCondensed-Bold.ttf",
        "BarlowCondensed-SemiBold.ttf",
        "PT_Sans-Narrow-Web-Bold.ttf",
        "FiraSansCondensed-Bold.ttf",
        "SairaCondensed-Bold.ttf",
        "EncodeSansCondensed-Bold.ttf",
    )]
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
_font_cache = {}


def _font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def find_code_box(bgr: np.ndarray):
    """Bounding box (x, y, w, h) of the black set-code box in a full card scan, or None."""
    H, W = bgr.shape[:2]
    y0, y1 = int(0.925 * H), int(0.975 * H)
    x0, x1 = int(0.07 * W), int(0.21 * W)
    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = (gray < 70).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.012 * H or h > 0.03 * H or not (1.6 <= w / h <= 3.6):
            continue
        if area < 0.45 * w * h:
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    if best is None:
        return None
    x, y, w, h, _ = best
    return (x0 + int(x), y0 + int(y), int(w), int(h))


def random_code(rng: np.random.Generator) -> str:
    if rng.random() < 0.12:
        # Codes like "30C": mix in digits.
        return "".join(rng.choice(list(LETTERS + DIGITS)) for _ in range(3))
    return "".join(rng.choice(list(LETTERS)) for _ in range(3))


def render_code_box(w: int, h: int, code: str, rng: np.random.Generator):
    """Returns (bgr, alpha) of size h x w for a box with the given code."""
    S = 6
    Wb, Hb = w * S, h * S
    img = Image.new("RGB", (Wb, Hb), (0, 0, 0))
    mask = Image.new("L", (Wb, Hb), 0)
    r = int(Hb * rng.uniform(0.15, 0.25))
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, Wb - 1, Hb - 1], radius=r, fill=255)
    d = ImageDraw.Draw(img)
    dark = int(rng.integers(0, 35))
    d.rounded_rectangle([0, 0, Wb - 1, Hb - 1], radius=r, fill=(dark, dark, dark))
    # Thin light outline just inside the edge.
    inset = Hb * rng.uniform(0.07, 0.12)
    lw = max(1, int(Hb * rng.uniform(0.04, 0.07)))
    light = int(rng.integers(150, 235))
    d.rounded_rectangle([inset, inset, Wb - 1 - inset, Hb - 1 - inset], radius=max(1, r - inset), outline=(light,) * 3, width=lw)

    font_path = FONTS[int(rng.integers(len(FONTS)))]
    cap_h = Hb * rng.uniform(0.52, 0.62)
    big = _font(font_path, int(cap_h / 0.72))
    small = _font(font_path, int(cap_h * rng.uniform(0.5, 0.6) / 0.72))
    # Render code and "EN" on their own layers so they can be squeezed to fit.
    def text_layer(text, font):
        bbox = font.getbbox(text)
        layer = Image.new("L", (bbox[2] - bbox[0] + 2, bbox[3] - bbox[1] + 2), 0)
        ImageDraw.Draw(layer).text((-bbox[0] + 1, -bbox[1] + 1), text, font=font, fill=255)
        return layer

    code_l = text_layer(code, big)
    en_l = text_layer("EN", small)
    target_code_w = Wb * rng.uniform(0.5, 0.6)
    target_en_w = Wb * rng.uniform(0.17, 0.22)
    code_l = code_l.resize((max(1, int(target_code_w)), max(1, int(cap_h))), Image.LANCZOS)
    en_h = cap_h * rng.uniform(0.5, 0.62)
    en_l = en_l.resize((max(1, int(target_en_w)), max(1, int(en_h))), Image.LANCZOS)
    left = Wb * rng.uniform(0.09, 0.14)
    gap = Wb * rng.uniform(0.02, 0.05)
    base = (Hb + cap_h) / 2 + Hb * rng.uniform(-0.04, 0.04)
    white = (int(rng.integers(215, 256)),) * 3
    img.paste(white, (int(left), int(base - cap_h)), code_l)
    img.paste(white, (int(left + target_code_w + gap), int(base - en_h)), en_l)

    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(np.asarray(mask), (w, h), interpolation=cv2.INTER_AREA)
    return bgr, alpha


# ---------------------------------------------------------------------------
# Synthetic collector numbers.
#
# Printed set totals ("/198") repeat across every card of a set, so the model
# also memorizes totals as words and misreads a new set's total. We erase the
# printed number (inpainting) and render a random one in a similar bold italic.

NUMBER_FONTS = _existing(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-BoldOblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-BoldItalic.ttf",
        "/usr/share/fonts/truetype/ibm-plex/IBMPlexSansCondensed-BoldItalic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
        EXTRA_FONTS / "FiraSansCondensed-BoldItalic.ttf",
        EXTRA_FONTS / "BarlowCondensed-BoldItalic.ttf",
    ]
)


def find_number_text(bgr: np.ndarray, box):
    """
    Locates the printed collector number right of the set-code box.
    Returns dict(x, y, w, h, light, color, mask) in full-scan pixels, or None.
    """
    if box is None:
        return None
    H, W = bgr.shape[:2]
    bx, by, bw, bh = box
    x0 = bx + bw + int(0.15 * bh)
    x1 = min(W, int(0.45 * W))
    y0 = max(0, by - int(0.15 * bh))
    y1 = min(H, by + bh + int(0.45 * bh))
    strip = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.int16)
    med = int(np.median(gray))
    light = med < 110
    if light:
        # White-on-art numbers have a dark outline that inpainting doesn't remove cleanly; skip them.
        return None
    mask = (gray > med + 70) if light else (gray < med - 60)
    mask = mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = sorted(
        (tuple(stats[i]) for i in range(1, n) if 0.4 * bh <= stats[i][3] <= 1.0 * bh and stats[i][4] >= 6),
        key=lambda s: s[0],
    )
    if not comps:
        return None
    # Glyphs from the left until a big gap (the rarity symbol follows after a gap).
    group = [comps[0]]
    for c in comps[1:]:
        prev = group[-1]
        if c[0] - (prev[0] + prev[2]) > 0.5 * bh:
            break
        group.append(c)
    gx0 = min(c[0] for c in group)
    gx1 = max(c[0] + c[2] for c in group)
    # Digit height/baseline from the median glyph ("/" can be taller).
    gy1 = int(np.median([c[1] + c[3] for c in group]))
    gy0 = gy1 - int(np.median([c[3] for c in group]))
    if gx1 - gx0 < 1.5 * bh or gx0 > 1.5 * bh or len(group) < 3:
        return None
    region = mask[gy0:gy1, gx0:gx1].astype(bool)
    color = strip[gy0:gy1, gx0:gx1][region].mean(axis=0) if region.any() else np.array([30, 30, 30])
    # Inpainting mask: only the grouped glyphs (plus a margin), not other shapes nearby.
    keep = np.zeros_like(mask)
    for c in group:
        keep[c[1] : c[1] + c[3], c[0] : c[0] + c[2]] |= mask[c[1] : c[1] + c[3], c[0] : c[0] + c[2]]
    my0 = min(c[1] for c in group)
    my1 = max(c[1] + c[3] for c in group)
    full_mask = np.zeros((H, W), np.uint8)
    full_mask[y0:y1, x0:x1] = keep * 255
    return {
        "x": x0 + gx0, "y": y0 + gy0, "w": gx1 - gx0, "h": gy1 - gy0,
        "light": bool(light), "color": tuple(int(v) for v in color),
        "mask_xy": (x0 + gx0 - 2, y0 + my0 - 2),
        "mask": full_mask[y0 + my0 - 2 : y0 + my1 + 2, x0 + gx0 - 2 : x0 + gx1 + 2],
    }


def random_number(rng: np.random.Generator, promo: bool) -> str:
    n = int(rng.integers(1, 320))
    if promo:
        return f"{n:03d}"
    return f"{n:03d}/{int(rng.integers(40, 320)):03d}"


def replace_number(img: np.ndarray, num: dict, old_text: str, new_text: str, rng: np.random.Generator) -> None:
    """Inpaints the old number away and draws `new_text` in its place (in place, BGR)."""
    x, y, w, h = num["x"], num["y"], num["w"], num["h"]
    x0, y0 = num["mask_xy"]
    mask = num["mask"]
    if x0 < 0 or y0 < 0:
        return
    roi = img[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
    if roi.shape[:2] != mask.shape:
        return
    m = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2 if num["light"] else 1)
    img[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = cv2.inpaint(roi, m, 3, cv2.INPAINT_TELEA)

    S = 6
    font_path = NUMBER_FONTS[int(rng.integers(len(NUMBER_FONTS)))]
    font = _font(font_path, int(h * S / 0.72))
    bbox = font.getbbox(new_text)
    stroke = int(h * S * 0.12) if num["light"] else 0
    tw, th = bbox[2] - bbox[0] + 2 * stroke + 4, bbox[3] - bbox[1] + 2 * stroke + 4
    fill_img = Image.new("RGB", (tw, th), (0, 0, 0))
    alpha = Image.new("L", (tw, th), 0)
    color = num["color"][::-1]  # BGR -> RGB
    fill = tuple(int(c) for c in color)
    stroke_fill = (20, 20, 20)
    pos = (-bbox[0] + stroke + 2, -bbox[1] + stroke + 2)
    ImageDraw.Draw(fill_img).text(pos, new_text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    ImageDraw.Draw(alpha).text(pos, new_text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    # Same glyph height as the original; width per character like the original.
    target_h = h + (2 * stroke // S)
    target_w = max(1, int(w * len(new_text) / max(1, len(old_text)) * rng.uniform(0.92, 1.08)))
    rgb = cv2.resize(np.asarray(fill_img), (target_w, target_h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(np.asarray(alpha), (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)[:, :, None] / 255.0
    ty = y + h - target_h + (stroke // S)
    tx = x
    H, W = img.shape[:2]
    if ty < 0 or tx < 0 or ty + target_h > H or tx + target_w > W:
        return
    region = img[ty : ty + target_h, tx : tx + target_w].astype(np.float32)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    img[ty : ty + target_h, tx : tx + target_w] = (bgr * a + region * (1 - a)).astype(np.uint8)
