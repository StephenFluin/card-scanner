import { GrayImage, otsuThreshold, resizeBilinearGray, RGBAImage, toGray } from './image';

export type Polarity = 'auto' | 'dark-text' | 'light-text';

export interface OcrPrepOptions {
  /** Height (px) to scale the crop to; Tesseract likes ~30px+ glyphs. */
  targetHeight: number;
  polarity: Polarity;
  /** Hard binarize with Otsu instead of handing Tesseract a contrast-stretched gray image. */
  binarize: boolean;
  /** White padding added on every side. */
  pad: number;
}

export interface OcrPrepResult {
  image: GrayImage;
  inverted: boolean;
}

function percentile(sorted: Float32Array, p: number): number {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))];
}

/**
 * Turns a rectified crop into a clean, upscaled, dark-text-on-white image for
 * Tesseract.
 */
export function prepareForOcr(crop: RGBAImage, opts: OcrPrepOptions): OcrPrepResult {
  let g = toGray(crop);
  const scale = opts.targetHeight / g.height;
  g = resizeBilinearGray(g, Math.max(1, Math.round(g.width * scale)), opts.targetHeight);

  // Contrast stretch between the 3rd and 97th percentiles.
  const sorted = Float32Array.from(g.data).sort();
  const lo = percentile(sorted, 0.03);
  const hi = percentile(sorted, 0.97);
  const range = Math.max(1, hi - lo);
  const data = new Float32Array(g.data.length);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.max(0, Math.min(255, ((g.data[i] - lo) / range) * 255));
  }
  g = { width: g.width, height: g.height, data };

  const t = otsuThreshold(g);
  let inverted = false;
  if (opts.polarity === 'light-text') {
    inverted = true;
  } else if (opts.polarity === 'auto') {
    // Text is the minority class: if most pixels are dark the text is light.
    let dark = 0;
    for (const v of g.data) if (v < t) dark++;
    inverted = dark > g.data.length / 2;
  }

  const out = new Float32Array(g.data.length);
  for (let i = 0; i < out.length; i++) {
    let v = inverted ? 255 - g.data[i] : g.data[i];
    if (opts.binarize) v = v < (inverted ? 255 - t : t) ? 0 : 255;
    out[i] = v;
  }
  return { image: pad({ width: g.width, height: g.height, data: out }, opts.pad), inverted };
}

function pad(g: GrayImage, p: number): GrayImage {
  if (p <= 0) return g;
  const w = g.width + 2 * p;
  const h = g.height + 2 * p;
  const data = new Float32Array(w * h).fill(255);
  for (let y = 0; y < g.height; y++) {
    data.set(g.data.subarray(y * g.width, (y + 1) * g.width), (y + p) * w + p);
  }
  return { width: w, height: h, data };
}

/** Encodes a gray image as binary PGM (P5), which Tesseract/Leptonica reads directly. */
export function encodePgm(g: GrayImage): Uint8Array {
  const header = new TextEncoder().encode(`P5\n${g.width} ${g.height}\n255\n`);
  const out = new Uint8Array(header.length + g.width * g.height);
  out.set(header, 0);
  for (let i = 0; i < g.data.length; i++) {
    out[header.length + i] = Math.max(0, Math.min(255, Math.round(g.data[i])));
  }
  return out;
}

/** Variance of the Laplacian: a cheap focus/sharpness measure. */
export function sharpness(img: RGBAImage): number {
  const g = toGray(img);
  const { width: w, height: h, data } = g;
  let sum = 0,
    sumSq = 0,
    n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const l = data[i - 1] + data[i + 1] + data[i - w] + data[i + w] - 4 * data[i];
      sum += l;
      sumSq += l * l;
      n++;
    }
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}
