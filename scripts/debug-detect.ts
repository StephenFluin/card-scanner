/**
 * Visualizes card detection on synthetic frames: edge points, candidate lines,
 * the chosen quad (green) and ground truth (red). Writes PNGs to .cache/detect/.
 *
 *   npx tsx scripts/debug-detect.ts [--n 12] [--seed 1] [--only-poor]
 */
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { lineEndpoints, Side } from '../src/app/vision/detect';
import type { Quad } from '../src/app/vision/homography';
import { Point, resizeArea, RGBAImage } from '../src/app/vision/image';
import { analyzeFrame, defaultGuide } from '../src/app/vision/pipeline';
import type { ScanLabel } from './download-scans';
import { loadRGBA, saveRGBA } from './lib/node-image';
import { mulberry32, randomParams, synthesize } from './lib/synth';

const ROOT = join(import.meta.dirname, '..');
const { values: args } = parseArgs({
  options: {
    n: { type: 'string', default: '12' },
    seed: { type: 'string', default: '1' },
    difficulty: { type: 'string', default: 'normal' },
    'only-poor': { type: 'boolean', default: false },
  },
});

function drawLine(img: RGBAImage, a: Point, b: Point, rgb: [number, number, number], w = 1): void {
  const n = Math.ceil(Math.hypot(b.x - a.x, b.y - a.y));
  for (let i = 0; i <= n; i++) {
    const x = a.x + ((b.x - a.x) * i) / n;
    const y = a.y + ((b.y - a.y) * i) / n;
    for (let dy = -w; dy <= w; dy++) {
      for (let dx = -w; dx <= w; dx++) {
        const px = Math.round(x + dx),
          py = Math.round(y + dy);
        if (px < 0 || py < 0 || px >= img.width || py >= img.height) continue;
        const p = (py * img.width + px) * 4;
        img.data[p] = rgb[0];
        img.data[p + 1] = rgb[1];
        img.data[p + 2] = rgb[2];
      }
    }
  }
}

function drawQuad(img: RGBAImage, q: Quad, rgb: [number, number, number], w: number): void {
  for (let i = 0; i < 4; i++) drawLine(img, q[i], q[(i + 1) % 4], rgb, w);
}

const COLORS: Record<Side, [number, number, number]> = {
  left: [255, 80, 80],
  right: [80, 160, 255],
  top: [255, 220, 40],
  bottom: [190, 140, 255],
};

async function main(): Promise<void> {
  const scans = JSON.parse(
    await readFile(join(ROOT, '.cache', 'scans', 'manifest.json'), 'utf8'),
  ) as ScanLabel[];
  const rand = mulberry32(parseInt(args.seed!, 10));
  const out = join(ROOT, '.cache', 'detect');
  await mkdir(out, { recursive: true });
  const frameW = 1080,
    frameH = 1920;
  const guide = defaultGuide(frameW, frameH);
  for (let i = 0; i < parseInt(args.n!, 10); i++) {
    const card = scans[Math.floor(rand() * scans.length)];
    const img = await loadRGBA(join(ROOT, '.cache', 'scans', card.file));
    const params = randomParams(rand, args.difficulty as 'normal');
    const { frame, quad: truth } = await synthesize(img, frameW, frameH, guide, params, rand);
    const a = analyzeFrame(frame, { guide, debug: true });
    const err = a.quad
      ? (Math.max(...a.quad.map((p, k) => Math.hypot(p.x - truth[k].x, p.y - truth[k].y))) /
          guide.w) *
        100
      : Infinity;
    if (args['only-poor'] && err < 1.5) continue;
    const vis: RGBAImage = {
      width: frame.width,
      height: frame.height,
      data: new Uint8ClampedArray(frame.data),
    };
    if (a.debug) {
      for (const side of ['left', 'right', 'top', 'bottom'] as Side[]) {
        const vertical = side === 'left' || side === 'right';
        for (const p of a.debug.points[side]) {
          const pt = vertical ? { x: p.s, y: p.t } : { x: p.t, y: p.s };
          drawLine(vis, pt, pt, COLORS[side], 2);
        }
        for (const l of a.debug.candidates[side]) {
          const [p, q] = lineEndpoints(l, frame.width, frame.height);
          drawLine(vis, p, q, COLORS[side], 0);
        }
      }
    }
    drawQuad(vis, truth, [255, 0, 0], 2);
    if (a.quad) drawQuad(vis, a.quad, [0, 255, 0], 1);
    const file = join(out, `${i}-${card.code}-${card.number}-err${err.toFixed(1)}.png`);
    await saveRGBA(resizeArea(vis, 540, 960), file);
    console.log(
      `${file}  score=${a.scores?.total.toFixed(2)} support=${a.scores?.support.toFixed(2)} aspect=${a.scores?.aspect.toFixed(2)} border=${a.scores?.border.toFixed(2)} (σ=${a.scores?.borderStd.toFixed(0)}) bg=${params.background} rot=${params.rotationDeg.toFixed(1)}`,
    );
    for (const side of ['left', 'right', 'top', 'bottom'] as Side[]) {
      console.log(
        `   ${side.padEnd(6)} ${a.debug?.candidates[side].map((l) => `b=${l.b.toFixed(0)} a=${l.a.toFixed(3)} sup=${l.support.toFixed(2)} str=${l.strength.toFixed(0)}`).join(' | ')}`,
      );
    }
  }
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
