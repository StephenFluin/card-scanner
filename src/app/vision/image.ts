/**
 * Minimal image primitives that work on plain RGBA buffers. Kept free of DOM
 * APIs so the same code runs in the browser, in a Web Worker and in Node
 * (offline evaluation scripts).
 */

export interface RGBAImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
}

export interface GrayImage {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
}

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export function createRGBA(width: number, height: number): RGBAImage {
  return { width, height, data: new Uint8ClampedArray(width * height * 4) };
}

export function toGray(img: RGBAImage): GrayImage {
  const out = new Float32Array(img.width * img.height);
  const d = img.data;
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  }
  return { width: img.width, height: img.height, data: out };
}

export function grayToRGBA(g: GrayImage): RGBAImage {
  const out = createRGBA(g.width, g.height);
  for (let i = 0, p = 0; i < g.data.length; i++, p += 4) {
    const v = g.data[i];
    out.data[p] = out.data[p + 1] = out.data[p + 2] = v;
    out.data[p + 3] = 255;
  }
  return out;
}

/** Box-filter (area average) resize. Good quality when shrinking. */
export function resizeArea(img: RGBAImage, outW: number, outH: number): RGBAImage {
  const out = createRGBA(outW, outH);
  const sx = img.width / outW;
  const sy = img.height / outH;
  const src = img.data;
  for (let oy = 0; oy < outH; oy++) {
    const y0 = oy * sy;
    const y1 = y0 + sy;
    const iy0 = Math.floor(y0);
    const iy1 = Math.min(img.height, Math.ceil(y1));
    for (let ox = 0; ox < outW; ox++) {
      const x0 = ox * sx;
      const x1 = x0 + sx;
      const ix0 = Math.floor(x0);
      const ix1 = Math.min(img.width, Math.ceil(x1));
      let r = 0, g = 0, b = 0, a = 0, wsum = 0;
      for (let y = iy0; y < iy1; y++) {
        const wy = Math.min(y + 1, y1) - Math.max(y, y0);
        for (let x = ix0; x < ix1; x++) {
          const w = wy * (Math.min(x + 1, x1) - Math.max(x, x0));
          const p = (y * img.width + x) * 4;
          r += src[p] * w;
          g += src[p + 1] * w;
          b += src[p + 2] * w;
          a += src[p + 3] * w;
          wsum += w;
        }
      }
      const q = (oy * outW + ox) * 4;
      out.data[q] = r / wsum;
      out.data[q + 1] = g / wsum;
      out.data[q + 2] = b / wsum;
      out.data[q + 3] = a / wsum;
    }
  }
  return out;
}

/** Bilinear resize, used for enlarging small crops before OCR. */
export function resizeBilinearGray(img: GrayImage, outW: number, outH: number): GrayImage {
  const out = new Float32Array(outW * outH);
  const sx = img.width / outW;
  const sy = img.height / outH;
  for (let oy = 0; oy < outH; oy++) {
    const fy = Math.max(0, Math.min(img.height - 1, (oy + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(img.height - 1, y0 + 1);
    const wy = fy - y0;
    for (let ox = 0; ox < outW; ox++) {
      const fx = Math.max(0, Math.min(img.width - 1, (ox + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(img.width - 1, x0 + 1);
      const wx = fx - x0;
      const a = img.data[y0 * img.width + x0];
      const b = img.data[y0 * img.width + x1];
      const c = img.data[y1 * img.width + x0];
      const d = img.data[y1 * img.width + x1];
      out[oy * outW + ox] = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy;
    }
  }
  return { width: outW, height: outH, data: out };
}

export function crop(img: RGBAImage, r: Rect): RGBAImage {
  const x = Math.max(0, Math.round(r.x));
  const y = Math.max(0, Math.round(r.y));
  const w = Math.min(img.width - x, Math.round(r.w));
  const h = Math.min(img.height - y, Math.round(r.h));
  const out = createRGBA(w, h);
  for (let row = 0; row < h; row++) {
    const start = ((y + row) * img.width + x) * 4;
    out.data.set(img.data.subarray(start, start + w * 4), row * w * 4);
  }
  return out;
}

export function rotate180(img: RGBAImage): RGBAImage {
  const out = createRGBA(img.width, img.height);
  const n = img.width * img.height;
  for (let i = 0; i < n; i++) {
    const p = i * 4;
    const q = (n - 1 - i) * 4;
    out.data[q] = img.data[p];
    out.data[q + 1] = img.data[p + 1];
    out.data[q + 2] = img.data[p + 2];
    out.data[q + 3] = img.data[p + 3];
  }
  return out;
}

/** Otsu threshold on a grayscale image (values 0..255). */
export function otsuThreshold(g: GrayImage): number {
  const hist = new Float64Array(256);
  for (const v of g.data) hist[Math.max(0, Math.min(255, v | 0))]++;
  const total = g.data.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, best = 0, threshold = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      threshold = t;
    }
  }
  return threshold;
}
