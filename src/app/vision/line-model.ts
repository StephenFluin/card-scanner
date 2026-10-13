import type { CardRegion } from './card-layout';
import type { RGBAImage } from './image';

/**
 * Pure TypeScript runtime for the info-line recognition model trained in
 * training/ (a small fully-convolutional CTC network). Reads e.g.
 * "SVI 001/198" from the rectified bottom-left line of a card.
 * No ML library needed: the network is only conv + ReLU + max-pool.
 */

export type LayerSpec =
  | { type: 'conv'; inC: number; outC: number; kh: number; kw: number; ph: number; pw: number; relu: boolean; w: number; b: number }
  | { type: 'maxpool'; kh: number; kw: number };

export interface LineModelManifest {
  name: string;
  createdAt: string;
  note?: string;
  input: { height: number; width: number; region: CardRegion; normalization: string };
  charset: string;
  dtype: 'float16' | 'float32';
  weights: string;
  params: number;
  layers: LayerSpec[];
}

export interface LineModelResult {
  text: string;
  /** Lowest per-character probability of the decoded text (0..1). */
  confidence: number;
  charConfidences: number[];
  ms: number;
}

interface Tensor {
  c: number;
  h: number;
  w: number;
  data: Float32Array;
}

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

type ConvSpec = Extract<LayerSpec, { type: 'conv' }>;

function conv(x: Tensor, L: ConvSpec, weights: Float32Array): Tensor {
  const oh = x.h + 2 * L.ph - L.kh + 1;
  const ow = x.w + 2 * L.pw - L.kw + 1;
  const out = new Float32Array(L.outC * oh * ow);
  const plane = oh * ow;
  // Four output channels per pass: each input value is loaded once for all four.
  const blocked = L.outC - (L.outC % 4);
  for (let o = 0; o < blocked; o += 4) convBlock4(x, L, weights, out, o, oh, ow);
  for (let o = blocked; o < L.outC; o++) convSingle(x, L, weights, out, o, oh, ow);
  if (L.relu) {
    for (let k = 0; k < out.length; k++) if (out[k] < 0) out[k] = 0;
  }
  return { c: L.outC, h: oh, w: ow, data: out };
}

function convBlock4(x: Tensor, L: ConvSpec, weights: Float32Array, out: Float32Array, o: number, oh: number, ow: number): void {
  const plane = oh * ow;
  const p0 = o * plane, p1 = p0 + plane, p2 = p1 + plane, p3 = p2 + plane;
  out.fill(weights[L.b + o], p0, p0 + plane);
  out.fill(weights[L.b + o + 1], p1, p1 + plane);
  out.fill(weights[L.b + o + 2], p2, p2 + plane);
  out.fill(weights[L.b + o + 3], p3, p3 + plane);
  const kk = L.kh * L.kw;
  const stride = L.inC * kk;
  const xd = x.data;
  for (let i = 0; i < L.inC; i++) {
    const ibase = i * x.h * x.w;
    for (let ky = 0; ky < L.kh; ky++) {
      for (let kx = 0; kx < L.kw; kx++) {
        const wi = L.w + (o * L.inC + i) * kk + ky * L.kw + kx;
        const w0 = weights[wi], w1 = weights[wi + stride], w2 = weights[wi + 2 * stride], w3 = weights[wi + 3 * stride];
        const dx = kx - L.pw;
        const xStart = Math.max(0, -dx);
        const xEnd = Math.min(ow, x.w - dx);
        for (let y = 0; y < oh; y++) {
          const sy = y + ky - L.ph;
          if (sy < 0 || sy >= x.h) continue;
          const orow = y * ow;
          const irow = ibase + sy * x.w + dx;
          for (let xx = xStart; xx < xEnd; xx++) {
            const v = xd[irow + xx];
            const k = orow + xx;
            out[p0 + k] += w0 * v;
            out[p1 + k] += w1 * v;
            out[p2 + k] += w2 * v;
            out[p3 + k] += w3 * v;
          }
        }
      }
    }
  }
}

function convSingle(x: Tensor, L: ConvSpec, weights: Float32Array, out: Float32Array, o: number, oh: number, ow: number): void {
  const plane = oh * ow;
  {
    const bias = weights[L.b + o];
    const obase = o * plane;
    out.fill(bias, obase, obase + plane);
    for (let i = 0; i < L.inC; i++) {
      const ibase = i * x.h * x.w;
      for (let ky = 0; ky < L.kh; ky++) {
        for (let kx = 0; kx < L.kw; kx++) {
          const wv = weights[L.w + ((o * L.inC + i) * L.kh + ky) * L.kw + kx];
          if (wv === 0) continue;
          const dx = kx - L.pw;
          const xStart = Math.max(0, -dx);
          const xEnd = Math.min(ow, x.w - dx);
          for (let y = 0; y < oh; y++) {
            const sy = y + ky - L.ph;
            if (sy < 0 || sy >= x.h) continue;
            const orow = obase + y * ow;
            const irow = ibase + sy * x.w + dx;
            for (let xx = xStart; xx < xEnd; xx++) out[orow + xx] += wv * x.data[irow + xx];
          }
        }
      }
    }
  }
}

function maxpool(x: Tensor, kh: number, kw: number): Tensor {
  const oh = Math.floor(x.h / kh);
  const ow = Math.floor(x.w / kw);
  const out = new Float32Array(x.c * oh * ow);
  for (let c = 0; c < x.c; c++) {
    for (let y = 0; y < oh; y++) {
      for (let xx = 0; xx < ow; xx++) {
        let m = -Infinity;
        for (let a = 0; a < kh; a++) {
          for (let b = 0; b < kw; b++) {
            const v = x.data[(c * x.h + y * kh + a) * x.w + xx * kw + b];
            if (v > m) m = v;
          }
        }
        out[(c * oh + y) * ow + xx] = m;
      }
    }
  }
  return { c: x.c, h: oh, w: ow, data: out };
}

/** numpy.percentile with linear interpolation, on a sorted array. */
function percentileSorted(sorted: Float32Array, p: number): number {
  const pos = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export class LineModel {
  private constructor(
    readonly manifest: LineModelManifest,
    private readonly weights: Float32Array,
  ) {}

  static fromBuffers(manifest: LineModelManifest, buffer: ArrayBuffer): LineModel {
    let weights: Float32Array;
    if (manifest.dtype === 'float16') {
      const half = new Uint16Array(buffer);
      weights = new Float32Array(half.length);
      for (let i = 0; i < half.length; i++) weights[i] = halfToFloat(half[i]);
    } else {
      weights = new Float32Array(buffer);
    }
    if (weights.length !== manifest.params) {
      throw new Error(`model weights: expected ${manifest.params} values, got ${weights.length}`);
    }
    return new LineModel(manifest, weights);
  }

  static async load(manifestUrl: string): Promise<LineModel> {
    const manifest = (await (await fetch(manifestUrl)).json()) as LineModelManifest;
    const weightsUrl = new URL(manifest.weights, new URL(manifestUrl, globalThis.location?.href)).toString();
    const buffer = await (await fetch(weightsUrl)).arrayBuffer();
    return LineModel.fromBuffers(manifest, buffer);
  }

  get region(): CardRegion {
    return this.manifest.input.region;
  }

  get inputSize(): { width: number; height: number } {
    return { width: this.manifest.input.width, height: this.manifest.input.height };
  }

  /**
   * Converts a rectified crop that is already exactly the model input size
   * into the normalized input tensor (gray, 2-98 percentile stretch, centred).
   */
  prepare(crop: RGBAImage): Float32Array {
    const { width, height } = this.inputSize;
    if (crop.width !== width || crop.height !== height) {
      throw new Error(`line model expects ${width}x${height}, got ${crop.width}x${crop.height}`);
    }
    const n = width * height;
    const g = new Float32Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      g[i] = 0.299 * crop.data[p] + 0.587 * crop.data[p + 1] + 0.114 * crop.data[p + 2];
    }
    const sorted = Float32Array.from(g).sort();
    const lo = percentileSorted(sorted, 2);
    const hi = percentileSorted(sorted, 98);
    const range = Math.max(hi - lo, 1);
    for (let i = 0; i < n; i++) g[i] = Math.min(1, Math.max(0, (g[i] - lo) / range)) - 0.5;
    return g;
  }

  run(input: Float32Array): LineModelResult {
    const start = performance.now();
    let x: Tensor = { c: 1, h: this.manifest.input.height, w: this.manifest.input.width, data: input };
    for (const L of this.manifest.layers) {
      x = L.type === 'conv' ? conv(x, L, this.weights) : maxpool(x, L.kh, L.kw);
    }
    // x: classes x 1 x T. Greedy CTC decode with softmax probabilities.
    const T = x.w;
    const C = x.c;
    const charset = this.manifest.charset;
    let text = '';
    const confs: number[] = [];
    let prev = 0;
    let runConf = 0;
    for (let t = 0; t < T; t++) {
      let best = 0;
      let bestV = -Infinity;
      let max = -Infinity;
      for (let c = 0; c < C; c++) max = Math.max(max, x.data[c * T + t]);
      let sum = 0;
      for (let c = 0; c < C; c++) {
        const v = x.data[c * T + t];
        sum += Math.exp(v - max);
        if (v > bestV) {
          bestV = v;
          best = c;
        }
      }
      const p = Math.exp(bestV - max) / sum;
      if (best !== 0 && best !== prev) {
        text += charset[best - 1];
        confs.push(p);
        runConf = p;
      } else if (best !== 0 && best === prev) {
        // Same character continuing: keep the most confident frame.
        runConf = Math.max(runConf, p);
        confs[confs.length - 1] = runConf;
      }
      prev = best;
    }
    return {
      text,
      confidence: confs.length ? Math.min(...confs) : 0,
      charConfidences: confs,
      ms: performance.now() - start,
    };
  }
}

export interface ParsedInfoLine {
  setCode: string;
  number: string;
  total: string | null;
}

/** "SVI 001/198" -> fields. Tolerates a missing space. */
export function parseInfoLine(text: string): ParsedInfoLine | null {
  const m = /^\s*([A-Z0-9]{3})\s*(\d{3})(?:\/(\d{3}))?\s*$/.exec(text);
  if (!m) return null;
  return { setCode: m[1], number: m[2], total: m[3] ?? null };
}
