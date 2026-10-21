/**
 * Synthesizes camera-like frames of a card from an official scan, with a known
 * ground-truth quad. Used by the offline evaluation and as a training data
 * generator.
 */
import { CARD_ASPECT } from '../../src/app/vision/detect';
import { applyHomography, homographyFromPoints, Quad } from '../../src/app/vision/homography';
import { createRGBA, Point, Rect, RGBAImage } from '../../src/app/vision/image';
import { gaussianBlur, jpegRoundTrip } from './node-image';

export type Rand = () => number;

export function mulberry32(seed: number): Rand {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const between = (r: Rand, a: number, b: number) => a + (b - a) * r();

export interface SynthParams {
  background: 'dark' | 'wood' | 'white' | 'clutter' | 'gray';
  scale: number;
  offsetX: number;
  offsetY: number;
  rotationDeg: number;
  /** Out-of-plane tilt of the card (degrees) around the horizontal and vertical axes. */
  tiltX: number;
  tiltY: number;
  /** Focal length as a multiple of the frame width. */
  focal: number;
  blur: number;
  noise: number;
  jpegQuality: number;
  brightness: number;
  colorCast: [number, number, number];
  glare: boolean;
  gradient: number;
}

export function randomParams(r: Rand, difficulty: 'easy' | 'normal' | 'hard'): SynthParams {
  const k = difficulty === 'easy' ? 0.4 : difficulty === 'normal' ? 1 : 1.6;
  const bgs: SynthParams['background'][] = ['dark', 'wood', 'white', 'clutter', 'gray'];
  return {
    background: bgs[Math.floor(r() * bgs.length)],
    scale: between(r, 1 - 0.14 * k, 1 + 0.04 * k),
    offsetX: between(r, -0.05, 0.05) * k,
    offsetY: between(r, -0.04, 0.04) * k,
    rotationDeg: between(r, -6, 6) * k,
    tiltX: between(r, -10, 10) * k,
    tiltY: between(r, -10, 10) * k,
    focal: between(r, 0.9, 1.4),
    blur: between(r, 0.3, 1.6) * k,
    noise: between(r, 2, 6) * k,
    jpegQuality: Math.round(between(r, 92, 75 - 10 * (k - 1))),
    brightness: between(r, 1 - 0.25 * k, 1 + 0.12 * k),
    colorCast: [between(r, 0.92, 1.08), between(r, 0.95, 1.05), between(r, 0.88, 1.08)],
    glare: r() < 0.25 * k,
    gradient: between(r, 0, 0.25) * k,
  };
}

function fillBackground(img: RGBAImage, kind: SynthParams['background'], r: Rand): void {
  const { width: w, height: h, data } = img;
  const base: Record<SynthParams['background'], [number, number, number]> = {
    dark: [28, 30, 34],
    wood: [120, 80, 50],
    white: [232, 230, 226],
    clutter: [90, 100, 110],
    gray: [150, 150, 150],
  };
  const [br, bg, bb] = base[kind];
  // Low-frequency blotches plus grain.
  const freq = between(r, 0.004, 0.02);
  const phase = r() * 100;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      if (kind === 'wood') v = 18 * Math.sin((y + 40 * Math.sin(x * 0.01 + phase)) * 0.05);
      else v = 10 * Math.sin(x * freq + phase) * Math.cos(y * freq * 0.7 + phase);
      const grain = (r() - 0.5) * 8;
      const p = (y * w + x) * 4;
      data[p] = br + v + grain;
      data[p + 1] = bg + v * 0.8 + grain;
      data[p + 2] = bb + v * 0.6 + grain;
      data[p + 3] = 255;
    }
  }
  if (kind === 'clutter') {
    for (let i = 0; i < 25; i++) {
      const rx = r() * w,
        ry = r() * h,
        rw = between(r, 30, 300),
        rh = between(r, 30, 300);
      const c = [r() * 255, r() * 255, r() * 255];
      for (let y = Math.max(0, ry | 0); y < Math.min(h, ry + rh); y++) {
        for (let x = Math.max(0, rx | 0); x < Math.min(w, rx + rw); x++) {
          const p = (y * w + x) * 4;
          data[p] = c[0];
          data[p + 1] = c[1];
          data[p + 2] = c[2];
        }
      }
    }
  }
}

function sampleBilinear(img: RGBAImage, x: number, y: number, out: number[]): void {
  const fx = Math.max(0, Math.min(img.width - 1.001, x - 0.5));
  const fy = Math.max(0, Math.min(img.height - 1.001, y - 0.5));
  const x0 = fx | 0,
    y0 = fy | 0;
  const wx = fx - x0,
    wy = fy - y0;
  for (let c = 0; c < 4; c++) {
    const a = img.data[(y0 * img.width + x0) * 4 + c];
    const b = img.data[(y0 * img.width + x0 + 1) * 4 + c];
    const d = img.data[((y0 + 1) * img.width + x0) * 4 + c];
    const e = img.data[((y0 + 1) * img.width + x0 + 1) * 4 + c];
    out[c] = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
  }
}

export interface SynthFrame {
  frame: RGBAImage;
  quad: Quad;
}

/** Renders `card` (an official scan) into a frame inside/around `guide`. */
export async function synthesize(
  card: RGBAImage,
  frameW: number,
  frameH: number,
  guide: Rect,
  p: SynthParams,
  r: Rand,
): Promise<SynthFrame> {
  const frame = createRGBA(frameW, frameH);
  fillBackground(frame, p.background, r);

  const cw = guide.w * p.scale;
  const ch = cw / CARD_ASPECT;
  const cx = guide.x + guide.w / 2 + p.offsetX * guide.w;
  const cy = guide.y + guide.h / 2 + p.offsetY * guide.h;
  // Real perspective: rotate the card in 3D (in-plane, then tilt about x and y)
  // and project with a pinhole camera placed so the card is ~cw wide.
  const f = p.focal * frameW;
  const g = (p.rotationDeg * Math.PI) / 180;
  const ax = (p.tiltX * Math.PI) / 180;
  const ay = (p.tiltY * Math.PI) / 180;
  const corners: Point[] = [
    { x: -cw / 2, y: -ch / 2 },
    { x: cw / 2, y: -ch / 2 },
    { x: cw / 2, y: ch / 2 },
    { x: -cw / 2, y: ch / 2 },
  ].map((c) => {
    let x = c.x * Math.cos(g) - c.y * Math.sin(g);
    let y = c.x * Math.sin(g) + c.y * Math.cos(g);
    let z = 0;
    // Tilt about the x axis.
    [y, z] = [y * Math.cos(ax) - z * Math.sin(ax), y * Math.sin(ax) + z * Math.cos(ax)];
    // Tilt about the y axis.
    [x, z] = [x * Math.cos(ay) + z * Math.sin(ay), -x * Math.sin(ay) + z * Math.cos(ay)];
    const k = f / (f + z);
    return { x: cx + x * k, y: cy + y * k };
  });
  const quad = corners as unknown as Quad;
  const toCard = homographyFromPoints(quad, [
    { x: 0, y: 0 },
    { x: card.width, y: 0 },
    { x: card.width, y: card.height },
    { x: 0, y: card.height },
  ]);
  const minX = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.x))));
  const maxX = Math.min(frameW - 1, Math.ceil(Math.max(...corners.map((c) => c.x))));
  const minY = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.y))));
  const maxY = Math.min(frameH - 1, Math.ceil(Math.max(...corners.map((c) => c.y))));
  const px = [0, 0, 0, 0];
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const s = applyHomography(toCard, { x: x + 0.5, y: y + 0.5 });
      if (s.x < 0 || s.y < 0 || s.x >= card.width || s.y >= card.height) continue;
      sampleBilinear(card, s.x, s.y, px);
      const alpha = px[3] / 255;
      const q = (y * frameW + x) * 4;
      for (let c = 0; c < 3; c++)
        frame.data[q + c] = frame.data[q + c] * (1 - alpha) + px[c] * alpha;
    }
  }

  // Lighting: global brightness, colour cast, linear gradient, optional glare.
  const gAngle = r() * Math.PI * 2;
  const gx = Math.cos(gAngle),
    gy = Math.sin(gAngle);
  const glareX = cx + (r() - 0.5) * cw,
    glareY = cy + (r() - 0.5) * ch,
    glareR = between(r, 0.1, 0.3) * cw;
  for (let y = 0; y < frameH; y++) {
    for (let x = 0; x < frameW; x++) {
      const q = (y * frameW + x) * 4;
      const grad = 1 + p.gradient * (((x - frameW / 2) * gx + (y - frameH / 2) * gy) / frameH);
      let glare = 0;
      if (p.glare) {
        const d2 = ((x - glareX) ** 2 + (y - glareY) ** 2) / (glareR * glareR);
        glare = 200 * Math.exp(-d2);
      }
      for (let c = 0; c < 3; c++) {
        frame.data[q + c] =
          frame.data[q + c] * p.brightness * p.colorCast[c] * grad +
          glare +
          (r() - 0.5) * 2 * p.noise;
      }
    }
  }
  const blurred = await gaussianBlur(frame, p.blur);
  return { frame: await jpegRoundTrip(blurred, p.jpegQuality), quad };
}
