import { applyHomography, cardToImage, Quad } from './homography';
import { Point, Rect, RGBAImage } from './image';

/**
 * Card edge detection.
 *
 * The user is asked to hold the card inside an on-screen guide, so we know
 * roughly where each edge should be. For each of the four sides we:
 *   1. scan perpendicular lines through a band around the expected edge and
 *      keep the strongest gradient peaks on each scanline,
 *   2. fit several candidate lines through those peaks with RANSAC,
 * then score every combination of one candidate per side. The score combines
 * how well the lines are supported, how close the quad is to the physical
 * 63x88mm aspect ratio, and whether there is a uniform strip just inside each
 * edge (the silver border on Scarlet & Violet / Mega Evolution era cards).
 * That last term is what lets us prefer the outer card edge over the strong
 * inner edge between the border and the card's coloured frame.
 */

export const CARD_ASPECT = 63 / 88;

export type Side = 'left' | 'right' | 'top' | 'bottom';
const SIDES: readonly Side[] = ['left', 'right', 'top', 'bottom'];

/**
 * A line in side-local coordinates: `s = a * t + b`, where `t` runs along the
 * edge and `s` across it. For left/right edges t=y and s=x, for top/bottom
 * edges t=x and s=y.
 */
export interface EdgeLine {
  side: Side;
  a: number;
  b: number;
  /** Fraction of scanlines that have a gradient peak on this line. */
  support: number;
  /** Mean gradient magnitude of the inlier peaks. */
  strength: number;
}

export interface EdgePoint {
  t: number;
  s: number;
  m: number;
}

export interface DetectOptions {
  /** Where the card is expected to be (the guide), in image pixels. */
  guide: Rect;
  /** Half-width of the edge search band as a fraction of the guide width. */
  band?: number;
  /** Minimum gradient magnitude for an edge peak (Sobel units on 0..255). */
  minGradient?: number;
  /** Collect per-side points and candidate lines for debugging. */
  debug?: boolean;
}

export interface DetectScores {
  total: number;
  support: number;
  aspect: number;
  border: number;
  /** 0..1: how often the chosen line is the outer line of a "card edge + inner frame edge" pair. */
  pair: number;
  /** 0..1: 1 when no other well-supported candidate line lies outside the chosen ones. */
  outer: number;
  aspectRatio: number;
  borderStd: number;
}

export interface DetectCandidate {
  quad: Quad;
  scores: DetectScores;
}

export interface DetectResult {
  quad: Quad | null;
  scores: DetectScores | null;
  /** The best few distinct quads, best first (includes `quad`). */
  alternatives: DetectCandidate[];
  debug?: {
    points: Record<Side, EdgePoint[]>;
    candidates: Record<Side, EdgeLine[]>;
  };
}

interface Gradient {
  width: number;
  height: number;
  gx: Float32Array;
  gy: Float32Array;
}

/**
 * Sobel gradient. For each pixel the colour channel with the largest
 * magnitude wins, so edges between similarly bright but differently coloured
 * regions (silver card on a wooden table, say) are still found.
 */
export function colorGradient(img: RGBAImage): Gradient {
  const { width: w, height: h, data: d } = img;
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const stride = w * 4;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * stride + x * 4;
      let bestM = -1, bx = 0, by = 0;
      for (let c = 0; c < 3; c++) {
        const tl = d[p - stride - 4 + c], t = d[p - stride + c], tr = d[p - stride + 4 + c];
        const l = d[p - 4 + c], r = d[p + 4 + c];
        const bl = d[p + stride - 4 + c], b = d[p + stride + c], br = d[p + stride + 4 + c];
        const sx = tr + 2 * r + br - tl - 2 * l - bl;
        const sy = bl + 2 * b + br - tl - 2 * t - tr;
        const m = sx * sx + sy * sy;
        if (m > bestM) {
          bestM = m;
          bx = sx;
          by = sy;
        }
      }
      gx[y * w + x] = bx;
      gy[y * w + x] = by;
    }
  }
  return { width: w, height: h, gx, gy };
}

function expectedEdge(side: Side, g: Rect): { s: number; t0: number; t1: number } {
  switch (side) {
    case 'left':
      return { s: g.x, t0: g.y, t1: g.y + g.h };
    case 'right':
      return { s: g.x + g.w, t0: g.y, t1: g.y + g.h };
    case 'top':
      return { s: g.y, t0: g.x, t1: g.x + g.w };
    case 'bottom':
      return { s: g.y + g.h, t0: g.x, t1: g.x + g.w };
  }
}

const PEAKS_PER_SCANLINE = 4;

function collectEdgePoints(
  grad: Gradient,
  side: Side,
  guide: Rect,
  bandPx: number,
  minGradient: number,
): { points: EdgePoint[]; scanlines: number } {
  const vertical = side === 'left' || side === 'right';
  const { s: s0, t0, t1 } = expectedEdge(side, guide);
  const len = t1 - t0;
  const tMax = vertical ? grad.height - 2 : grad.width - 2;
  const sMax = vertical ? grad.width - 2 : grad.height - 2;
  // Skip the rounded corners and leave room for some rotation.
  const tStart = Math.max(1, Math.round(t0 + len * 0.12));
  const tEnd = Math.min(tMax, Math.round(t1 - len * 0.12));
  const step = Math.max(1, Math.round(len / 90));
  const sStart = Math.max(1, Math.round(s0 - bandPx));
  const sEnd = Math.min(sMax, Math.round(s0 + bandPx));
  const points: EdgePoint[] = [];
  let scanlines = 0;
  const mags: number[] = [];
  for (let t = tStart; t <= tEnd; t += step) {
    scanlines++;
    mags.length = 0;
    for (let s = sStart; s <= sEnd; s++) {
      const idx = vertical ? t * grad.width + s : s * grad.width + t;
      const across = vertical ? grad.gx[idx] : grad.gy[idx];
      const along = vertical ? grad.gy[idx] : grad.gx[idx];
      const a = Math.abs(across);
      // The edge must be roughly parallel to the expected side (< ~30 deg).
      mags.push(a > minGradient && Math.abs(along) < a * 0.6 ? a : 0);
    }
    const peaks: EdgePoint[] = [];
    for (let i = 1; i < mags.length - 1; i++) {
      const m = mags[i];
      if (m > 0 && m >= mags[i - 1] && m > mags[i + 1]) {
        // Sub-pixel refinement with a parabola through the three samples.
        const denom = mags[i - 1] - 2 * m + mags[i + 1];
        const offset = denom !== 0 ? (0.5 * (mags[i - 1] - mags[i + 1])) / denom : 0;
        peaks.push({ t, s: sStart + i + offset, m });
      }
    }
    peaks.sort((p, q) => q.m - p.m);
    points.push(...peaks.slice(0, PEAKS_PER_SCANLINE));
  }
  return { points, scanlines };
}

/** Deterministic PRNG so detection is reproducible frame to frame. */
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fitLines(
  side: Side,
  points: EdgePoint[],
  scanlines: number,
  tol: number,
  maxLines: number,
): EdgeLine[] {
  const rand = mulberry32(1234);
  const lines: EdgeLine[] = [];
  let remaining = points;
  const maxSlope = 0.55; // ~29 degrees
  for (let k = 0; k < maxLines && remaining.length >= 2; k++) {
    let bestScore = 0;
    let bestA = 0, bestB = 0;
    const iterations = Math.min(400, remaining.length * 4);
    const tMin = remaining.reduce((m, p) => Math.min(m, p.t), Infinity);
    const tMaxV = remaining.reduce((m, p) => Math.max(m, p.t), -Infinity);
    const minSpan = (tMaxV - tMin) * 0.25;
    for (let i = 0; i < iterations; i++) {
      const p = remaining[(rand() * remaining.length) | 0];
      const q = remaining[(rand() * remaining.length) | 0];
      if (Math.abs(q.t - p.t) < minSpan) continue;
      const a = (q.s - p.s) / (q.t - p.t);
      if (Math.abs(a) > maxSlope) continue;
      const b = p.s - a * p.t;
      const score = scoreLine(remaining, a, b, tol);
      if (score > bestScore) {
        bestScore = score;
        bestA = a;
        bestB = b;
      }
    }
    if (bestScore === 0) break;
    // Weighted least-squares refinement on the inliers.
    let a = bestA, b = bestB;
    for (let iter = 0; iter < 2; iter++) {
      let sw = 0, st = 0, ss = 0, stt = 0, sts = 0;
      for (const p of remaining) {
        if (Math.abs(p.s - (a * p.t + b)) > tol) continue;
        const w = p.m;
        sw += w;
        st += w * p.t;
        ss += w * p.s;
        stt += w * p.t * p.t;
        sts += w * p.t * p.s;
      }
      const den = sw * stt - st * st;
      if (sw === 0 || Math.abs(den) < 1e-9) break;
      a = (sw * sts - st * ss) / den;
      b = (ss - a * st) / sw;
    }
    const inlierScanlines = new Set<number>();
    let strength = 0, n = 0;
    for (const p of remaining) {
      if (Math.abs(p.s - (a * p.t + b)) <= tol) {
        inlierScanlines.add(p.t);
        strength += p.m;
        n++;
      }
    }
    const support = inlierScanlines.size / scanlines;
    if (support < 0.25) break;
    lines.push({ side, a, b, support, strength: n ? strength / n : 0 });
    // Remove the inliers (with a little extra margin) and look for the next line.
    remaining = remaining.filter((p) => Math.abs(p.s - (a * p.t + b)) > tol * 2);
  }
  return lines;
}

function scoreLine(points: EdgePoint[], a: number, b: number, tol: number): number {
  // Count each scanline at most once so a cluster of peaks can't dominate.
  let score = 0;
  let lastT = Number.NaN;
  let bestOnScanline = 0;
  for (const p of points) {
    if (p.t !== lastT) {
      score += bestOnScanline;
      bestOnScanline = 0;
      lastT = p.t;
    }
    if (Math.abs(p.s - (a * p.t + b)) <= tol) {
      // Saturate the magnitude so one very strong edge doesn't outvote a long one.
      bestOnScanline = Math.max(bestOnScanline, 1 + Math.min(p.m, 400) / 400);
    }
  }
  return score + bestOnScanline;
}

/** Intersection of a vertical-ish (x = a1 y + b1) and horizontal-ish (y = a2 x + b2) line. */
function intersect(v: EdgeLine, h: EdgeLine): Point {
  const x = (v.a * h.b + v.b) / (1 - v.a * h.a);
  return { x, y: h.a * x + h.b };
}

function dist(p: Point, q: Point): number {
  return Math.hypot(p.x - q.x, p.y - q.y);
}

function isConvex(q: Quad): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (cross === 0) return false;
    const s = Math.sign(cross);
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}

function sampleLuma(img: RGBAImage, p: Point): number {
  const x = Math.max(0, Math.min(img.width - 1, Math.round(p.x)));
  const y = Math.max(0, Math.min(img.height - 1, Math.round(p.y)));
  const i = (y * img.width + x) * 4;
  return 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
}

/**
 * Measures how uniform a thin strip just inside each edge of the quad is.
 * Returns the mean (over sides) of the luminance standard deviation.
 */
export function borderUniformity(img: RGBAImage, quad: Quad): number {
  const H = cardToImage(quad);
  // The silver border is ~3.5% of the card width; sample in its middle.
  const du = 0.016;
  const dv = du * CARD_ASPECT;
  const strips: ((k: number) => Point)[] = [
    (k) => ({ x: du, y: k }),
    (k) => ({ x: 1 - du, y: k }),
    (k) => ({ x: k, y: dv }),
    (k) => ({ x: k, y: 1 - dv }),
  ];
  let total = 0;
  for (const strip of strips) {
    const vals: number[] = [];
    for (let i = 0; i <= 24; i++) {
      const k = 0.1 + (0.8 * i) / 24;
      vals.push(sampleLuma(img, applyHomography(H, strip(k))));
    }
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
    const variance = vals.reduce((s, v) => s + (v - mean) * (v - mean), 0) / vals.length;
    total += Math.sqrt(variance);
  }
  return total / 4;
}

/**
 * For every candidate line, 1 / (1 + number of well-supported candidates
 * further out). Card content is inside the card edge, so a strong line
 * outside the chosen one suggests we picked an internal line.
 */
function outerness(side: Side, lines: EdgeLine[], guide: Rect, tol: number): number[] {
  const { t0, t1 } = expectedEdge(side, guide);
  const tm = (t0 + t1) / 2;
  const outerSign = side === 'left' || side === 'top' ? -1 : 1;
  return lines.map((a) => {
    let n = 0;
    for (const b of lines) {
      if (b === a || b.support < 0.35) continue;
      const d = ((b.a * tm + b.b) - (a.a * tm + a.b)) * outerSign;
      if (d > tol * 2) n++;
    }
    return 1 / (1 + n);
  });
}

/** The silver border is ~3.3% of the card width; accept a generous range. */
const BORDER_MIN = 0.018;
const BORDER_MAX = 0.055;

/**
 * For every candidate line: +1 if it is the outer line of a parallel pair
 * separated by about the border width (card edge + inner frame edge), -1 if
 * it is the inner line of such a pair, 0 otherwise.
 */
function pairRoles(side: Side, lines: EdgeLine[], guide: Rect): number[] {
  const { t0, t1 } = expectedEdge(side, guide);
  const tm = (t0 + t1) / 2;
  const outerSign = side === 'left' || side === 'top' ? -1 : 1;
  const roles = lines.map(() => 0);
  for (let i = 0; i < lines.length; i++) {
    for (let j = 0; j < lines.length; j++) {
      if (i === j) continue;
      const a = lines[i], b = lines[j];
      if (Math.abs(a.a - b.a) > 0.04 || Math.min(a.support, b.support) < 0.35) continue;
      // Positive when line i is further out than line j.
      const d = ((a.a * tm + a.b) - (b.a * tm + b.b)) * outerSign;
      if (d >= BORDER_MIN * guide.w && d <= BORDER_MAX * guide.w) {
        roles[i] = Math.max(roles[i], 1);
        roles[j] = roles[j] === 1 ? 1 : -1;
      }
    }
  }
  return roles;
}

export function detectCard(img: RGBAImage, opts: DetectOptions): DetectResult {
  const guide = opts.guide;
  const bandPx = (opts.band ?? 0.22) * guide.w;
  const minGradient = opts.minGradient ?? 48;
  const grad = colorGradient(img);
  const tol = Math.max(1.2, guide.w / 220);

  const points = {} as Record<Side, EdgePoint[]>;
  const candidates = {} as Record<Side, EdgeLine[]>;
  for (const side of SIDES) {
    const { points: pts, scanlines } = collectEdgePoints(grad, side, guide, bandPx, minGradient);
    points[side] = pts;
    candidates[side] = fitLines(side, pts, scanlines, tol, 5);
  }
  const debug = opts.debug ? { points, candidates } : undefined;
  if (SIDES.some((s) => candidates[s].length === 0)) {
    return { quad: null, scores: null, alternatives: [], debug };
  }

  const outer = Object.fromEntries(SIDES.map((s) => [s, outerness(s, candidates[s], guide, tol)])) as Record<Side, number[]>;
  const roles = Object.fromEntries(SIDES.map((s) => [s, pairRoles(s, candidates[s], guide)])) as Record<Side, number[]>;
  const all: DetectCandidate[] = [];
  for (const [li, l] of candidates.left.entries()) {
    for (const [ri, r] of candidates.right.entries()) {
      for (const [ti, t] of candidates.top.entries()) {
        for (const [bi, b] of candidates.bottom.entries()) {
          const quad: Quad = [intersect(l, t), intersect(r, t), intersect(r, b), intersect(l, b)];
          if (!isConvex(quad)) continue;
          const w = (dist(quad[0], quad[1]) + dist(quad[3], quad[2])) / 2;
          const h = (dist(quad[0], quad[3]) + dist(quad[1], quad[2])) / 2;
          if (w < guide.w * 0.45 || h < guide.h * 0.45) continue;
          const aspectRatio = w / h;
          const aspect = Math.max(0, 1 - Math.abs(Math.log(aspectRatio / CARD_ASPECT)) / 0.07);
          if (aspect === 0) continue;
          const support = (l.support + r.support + t.support + b.support) / 4;
          const borderStd = borderUniformity(img, quad);
          const border = Math.max(0, 1 - borderStd / 30);
          const pair = (roles.left[li] + roles.right[ri] + roles.top[ti] + roles.bottom[bi] + 4) / 8;
          const out = (outer.left[li] + outer.right[ri] + outer.top[ti] + outer.bottom[bi]) / 4;
          const total = 0.4 * support + 0.2 * aspect + 0.05 * border + 0.2 * pair + 0.15 * out;
          all.push({ quad, scores: { total, support, aspect, border, pair, outer: out, aspectRatio, borderStd } });
        }
      }
    }
  }
  all.sort((a, b) => b.scores.total - a.scores.total);
  // Keep a few clearly different hypotheses for the reader to try.
  const alternatives: DetectCandidate[] = [];
  const minDiff = guide.w * 0.015;
  for (const c of all) {
    if (alternatives.length >= 4) break;
    if (alternatives.every((a) => Math.max(...a.quad.map((p, i) => dist(p, c.quad[i]))) > minDiff)) alternatives.push(c);
  }
  const best = alternatives[0] ?? null;
  return { quad: best?.quad ?? null, scores: best?.scores ?? null, alternatives, debug };
}

/** Converts a side-local line into two image points spanning the image, for drawing. */
export function lineEndpoints(line: EdgeLine, width: number, height: number): [Point, Point] {
  if (line.side === 'left' || line.side === 'right') {
    return [
      { x: line.b, y: 0 },
      { x: line.a * height + line.b, y: height },
    ];
  }
  return [
    { x: 0, y: line.b },
    { x: width, y: line.a * width + line.b },
  ];
}
