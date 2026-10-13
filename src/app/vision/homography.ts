import { createRGBA, Point, RGBAImage } from './image';

/** Row-major 3x3 matrix. */
export type Mat3 = readonly number[];

/** Solves Ax = b in place with Gaussian elimination and partial pivoting. */
function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
    }
    [A[col], A[pivot]] = [A[pivot], A[col]];
    [b[col], b[pivot]] = [b[pivot], b[col]];
    const p = A[col][col];
    if (Math.abs(p) < 1e-12) throw new Error('Singular homography system');
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / p;
      for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= A[r][c] * x[c];
    x[r] = s / A[r][r];
  }
  return x;
}

/** Homography mapping each `from[i]` onto `to[i]` (exactly four points). */
export function homographyFromPoints(from: readonly Point[], to: readonly Point[]): Mat3 {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = from[i];
    const { x: u, y: v } = to[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const h = solve(A, b);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

export function applyHomography(H: Mat3, p: Point): Point {
  const w = H[6] * p.x + H[7] * p.y + H[8];
  return { x: (H[0] * p.x + H[1] * p.y + H[2]) / w, y: (H[3] * p.x + H[4] * p.y + H[5]) / w };
}

/**
 * A card quad in source-image pixels, ordered top-left, top-right,
 * bottom-right, bottom-left (as the card is read, not as it appears on screen).
 */
export type Quad = readonly [Point, Point, Point, Point];

const UNIT_SQUARE: Quad = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

/** Homography from normalized card coordinates (0..1, 0..1) to source pixels. */
export function cardToImage(quad: Quad): Mat3 {
  return homographyFromPoints(UNIT_SQUARE, quad);
}

/** Rotates a quad's reading order by 180 degrees (card held upside down). */
export function rotateQuad180(q: Quad): Quad {
  return [q[2], q[3], q[0], q[1]];
}

/**
 * Renders a region of the card, given in normalized card coordinates, into a
 * new image. Uses `samples`x`samples` supersampling with bilinear taps so
 * shrinking a large source stays reasonably anti-aliased.
 */
export function warpCardRegion(
  src: RGBAImage,
  H: Mat3,
  region: { x: number; y: number; w: number; h: number },
  outW: number,
  outH: number,
  samples = 2,
): RGBAImage {
  const out = createRGBA(outW, outH);
  const sd = src.data;
  const sw = src.width;
  const sh = src.height;
  const inv = 1 / (samples * samples);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let r = 0, g = 0, b = 0;
      for (let sy = 0; sy < samples; sy++) {
        const v = region.y + ((oy + (sy + 0.5) / samples) / outH) * region.h;
        for (let sx = 0; sx < samples; sx++) {
          const u = region.x + ((ox + (sx + 0.5) / samples) / outW) * region.w;
          const w = H[6] * u + H[7] * v + H[8];
          let fx = (H[0] * u + H[1] * v + H[2]) / w - 0.5;
          let fy = (H[3] * u + H[4] * v + H[5]) / w - 0.5;
          fx = fx < 0 ? 0 : fx > sw - 1 ? sw - 1 : fx;
          fy = fy < 0 ? 0 : fy > sh - 1 ? sh - 1 : fy;
          const x0 = fx | 0;
          const y0 = fy | 0;
          const x1 = x0 + 1 < sw ? x0 + 1 : x0;
          const y1 = y0 + 1 < sh ? y0 + 1 : y0;
          const wx = fx - x0;
          const wy = fy - y0;
          const p00 = (y0 * sw + x0) * 4;
          const p10 = (y0 * sw + x1) * 4;
          const p01 = (y1 * sw + x0) * 4;
          const p11 = (y1 * sw + x1) * 4;
          const w00 = (1 - wx) * (1 - wy);
          const w10 = wx * (1 - wy);
          const w01 = (1 - wx) * wy;
          const w11 = wx * wy;
          r += sd[p00] * w00 + sd[p10] * w10 + sd[p01] * w01 + sd[p11] * w11;
          g += sd[p00 + 1] * w00 + sd[p10 + 1] * w10 + sd[p01 + 1] * w01 + sd[p11 + 1] * w11;
          b += sd[p00 + 2] * w00 + sd[p10 + 2] * w10 + sd[p01 + 2] * w01 + sd[p11 + 2] * w11;
        }
      }
      const q = (oy * outW + ox) * 4;
      out.data[q] = r * inv;
      out.data[q + 1] = g * inv;
      out.data[q + 2] = b * inv;
      out.data[q + 3] = 255;
    }
  }
  return out;
}

/**
 * Grows a quad outward by fractions of the card's width (dx) and height (dy),
 * measured in the card's own (perspective-corrected) coordinates.
 */
export function expandQuad(q: Quad, dx: number, dy: number): Quad {
  const H = cardToImage(q);
  return [
    applyHomography(H, { x: -dx, y: -dy }),
    applyHomography(H, { x: 1 + dx, y: -dy }),
    applyHomography(H, { x: 1 + dx, y: 1 + dy }),
    applyHomography(H, { x: -dx, y: 1 + dy }),
  ];
}
