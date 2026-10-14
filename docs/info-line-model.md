# Info-line model: reading a Pokémon card's set code and number

A small neural network that reads the **info line** printed in the bottom-left
corner of modern English Pokémon TCG cards and returns its text:

```
 [G] [SVI EN]  015/198 ★        →   "SVI 015/198"
 [H] [SVP EN]  050 ★            →   "SVP 050"        (promos have no total)
```

The set code plus the collector number uniquely identifies a card. Because the
model reads characters rather than memorizing cards, it also works on **sets it
has never seen**, including sets released after it was trained.

| | |
| --- | --- |
| Cards | Silver-border English cards: Scarlet & Violet and Mega Evolution series (2023+), including promos |
| Size | 227k parameters, 444 KB (float16) |
| Speed | ~25 ms per read in plain JavaScript on a desktop CPU; no ML library or GPU needed |
| License | AGPL-3.0-or-later (see [LICENSE](../LICENSE)) |

## Files

| File | What it is |
| --- | --- |
| [`public/models/info-line.json`](../public/models/info-line.json) | Manifest: input size and card region, character set, layer list |
| [`public/models/info-line.bin`](../public/models/info-line.bin) | Weights: little-endian float16, offsets given in the manifest |
| [`src/app/vision/line-model.ts`](../src/app/vision/line-model.ts) | TypeScript runtime (browser, Web Worker or Node), no dependencies |
| [`training/infer.py`](../training/infer.py) | Python reference implementation (NumPy + Pillow only) |
| [`training/`](../training/README.md) | Training code to reproduce or retrain the model |

## Quick start

### Python

```bash
pip install numpy pillow
python training/infer.py info-line-crop.png
# SVI 015/198	1.000
```

`infer.py` exposes `load()`, `prepare()` and `run()`, so you can also import it.

### TypeScript / JavaScript

```ts
import { LineModel, parseInfoLine } from './line-model';
import { cardToImage, warpCardRegion } from './homography';

const model = await LineModel.load('/models/info-line.json');

// `frame` is { width, height, data: Uint8ClampedArray } (RGBA, e.g. from getImageData).
// `corners` are the card's four corners in the frame, in reading order:
// top-left, top-right, bottom-right, bottom-left.
const H = cardToImage(corners);
const crop = warpCardRegion(frame, H, model.region, model.inputSize.width, model.inputSize.height, 2);

const { text, confidence } = model.run(model.prepare(crop));
const parsed = parseInfoLine(text); // { setCode: 'SVI', number: '015', total: '198' } or null
```

Copy `line-model.ts`, `homography.ts` and `image.ts` from `src/app/vision/`; they
have no dependencies. In Node, where there's no page URL to resolve against, load
with `LineModel.fromBuffers(manifestJson, weightsArrayBuffer)` instead of `load()`.

## Input: what to feed it

The model expects one specific region of the card, straightened, as a
**248 × 32** image. The region is given in *normalized card coordinates*
(0–1 across the full physical card, including the silver border) and is also
stored in the manifest (`input.region`):

```
x = 0.02   y = 0.934   width = 0.37   height = 0.034
```

How to get that crop:

1. **From a straight scan** (the card fills the image, upright): crop
   `x0 = 0.02·W, y0 = 0.934·H, x1 = 0.39·W, y1 = 0.968·H`, then resize to 248 × 32.
2. **From a camera photo**: find the card's four corners (any method; this repo's
   `detect.ts` is one), compute the homography from the unit square to those
   corners, and sample the region through it. `warpCardRegion` does this with
   2 × 2 supersampling, which matches how the model was trained.

Then normalize, as `prepare()` does in both implementations:

1. Grayscale: `0.299 R + 0.587 G + 0.114 B`
2. Contrast stretch: map the 2nd–98th percentile to 0–1 and clip
3. Subtract 0.5, giving a tensor of shape 1 × 32 × 248

The model tolerates the corner-detection errors of a typical phone camera
(about ±0.5% of the card width) plus blur, noise, JPEG artifacts, uneven
lighting and glare. If the card might be upside down, also try the crop
rotated by 180°: the model was trained to return an empty string for
upside-down or misplaced crops.

## Output: decoding

The network outputs 62 time steps × 39 classes (class 0 = CTC blank, classes
1–38 = the characters in `charset`: ` /0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`).
Take the most likely class per step, merge repeats, and drop blanks.

- **Text format:** `CODE NNN/TTT`, or `CODE NNN` for promos. Parse it with
  `^([A-Z0-9]{3})\s*(\d{3})(?:/(\d{3}))?$`. The regulation mark, the "EN"
  language tag and the rarity symbol are deliberately not output.
- **Confidence:** the lowest per-character probability. Reads below about 0.5
  are unreliable. An empty or unparseable result means "nothing readable here".

## Using it well

A single frame is right about 90% of the time on camera images. Combining
frames makes it much more reliable. In this app, accepting a card after **2
identical reads** (3 if the set is unknown) gives about 95% correct captures and
0% wrong ones in testing. Optionally, check the result against a list of known
sets and their printed totals (`SVI` → 198) to catch near-misses.

## Limitations

- **Silver-border English cards only.** Older yellow-border cards don't print a
  set code. Other languages and layouts weren't trained.
- **Trained on synthetic camera images** made from official card scans. It works
  well on real phones in practice, but there's no formal real-photo benchmark yet.
- **Characters never seen in a real set code** (currently N, Q, U, X, Y, Z and
  digits other than 0 and 3) were learned from similar fonts only, so a future
  code using them may need a retrain. See [training/README.md](../training/README.md).

## Manifest format

```jsonc
{
  "name": "info-line",
  "input": { "height": 32, "width": 248, "region": { "x": 0.02, "y": 0.934, "w": 0.37, "h": 0.034 },
             "normalization": "gray-p2-p98-centered" },
  "charset": " /0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  "dtype": "float16",
  "weights": "info-line.bin",
  "params": 227399,
  "layers": [
    { "type": "conv", "inC": 1, "outC": 16, "kh": 3, "kw": 3, "ph": 1, "pw": 1, "relu": true, "w": 0, "b": 144 },
    { "type": "maxpool", "kh": 2, "kw": 2 },
    ...
  ]
}
```

Layers run in order on a (channels, height, width) tensor. `conv` is a
stride-1 cross-correlation with zero padding `ph`/`pw`, batch-norm already folded
in, and weights at offset `w` (shape outC × inC × kh × kw, row-major) and bias at
offset `b` (outC values) in the weights file, counted in values, not bytes.
`maxpool` uses a stride equal to its kernel and drops any remainder. The final
tensor is classes × 1 × time steps.

## Retraining

[training/README.md](../training/README.md) covers the whole pipeline: download
scans, render synthetic training data, train (about 15 minutes on an RTX-class
GPU), and export to these two files.
