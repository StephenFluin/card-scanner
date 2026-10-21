/**
 * End-to-end offline evaluation of the scanning pipeline on synthetic camera
 * frames built from official card scans (see scripts/download-scans.ts):
 * frame synthesis -> card edge detection -> rectification -> info-line model
 * (-> optional Tesseract name read). Uses exactly the TypeScript code the app runs.
 *
 *   npm run eval -- [--n 100] [--seed 1] [--difficulty easy|normal|hard]
 *                   [--clean] [--names] [--sets PBL,DRI] [--save-failures]
 *
 * --clean skips synthesis and detection (crops straight from the scan).
 * --sets restricts to some set codes (e.g. the sets held out of training).
 */
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import {
  INFO_LINE,
  INFO_LINE_INPUT,
  NAME_CARD_WIDTH,
  NAME_REGION,
} from '../src/app/vision/card-layout';
import { Catalog, CatalogIndex, cardKey } from '../src/app/vision/catalog';
import { ReadFusion } from '../src/app/vision/fusion';
import { cardToImage, Quad, warpCardRegion } from '../src/app/vision/homography';
import type { RGBAImage } from '../src/app/vision/image';
import { LineModel, LineModelManifest } from '../src/app/vision/line-model';
import { nameSimilarity } from '../src/app/vision/parse';
import {
  analyzeFrame,
  cropRegion,
  defaultGuide,
  FrameCrops,
  readCard,
} from '../src/app/vision/pipeline';
import { InfoRead, readInfoLine, readName } from '../src/app/vision/reader';
import { TesseractEngine } from '../src/app/vision/tesseract-engine';
import type { ScanLabel } from './download-scans';
import { loadRGBA, saveGray, saveRGBA } from './lib/node-image';
import { mulberry32, randomParams, synthesize } from './lib/synth';

const ROOT = join(import.meta.dirname, '..');
const CACHE = join(ROOT, '.cache');

const { values: args } = parseArgs({
  options: {
    n: { type: 'string', default: '100' },
    seed: { type: 'string', default: '1' },
    difficulty: { type: 'string', default: 'normal' },
    clean: { type: 'boolean', default: false },
    names: { type: 'boolean', default: false },
    'save-failures': { type: 'boolean', default: false },
    sets: { type: 'string' },
    quiet: { type: 'boolean', default: false },
    'detect-only': { type: 'boolean', default: false },
    fusion: { type: 'string', default: '0' },
    'no-catalog': { type: 'boolean', default: false },
  },
});

function cornerError(a: Quad, b: Quad): number {
  return Math.max(...a.map((p, i) => Math.hypot(p.x - b[i].x, p.y - b[i].y)));
}

/**
 * Simulates holding one card in front of the camera: K consecutive frames of
 * the same scene (small pose/blur jitter, fresh noise) through the real
 * multi-frame fusion. Reports correct captures, wrong captures and misses.
 */
async function fusionEval(
  model: LineModel,
  index: CatalogIndex | null,
  picks: ScanLabel[],
  k: number,
): Promise<void> {
  const rand = mulberry32(parseInt(args.seed!, 10) + 1000);
  const guide = defaultGuide(1080, 1920);
  const infoLine = { region: model.region, ...model.inputSize };
  let correct = 0,
    wrong = 0,
    none = 0,
    framesToConfirm = 0;
  for (const card of picks) {
    const img = await loadRGBA(join(CACHE, 'scans', card.file));
    const base = randomParams(rand, args.difficulty as 'easy' | 'normal' | 'hard');
    const fusion = new ReadFusion(index);
    let outcome = 'none';
    let t = 0;
    for (let f = 0; f < k; f++) {
      const p = {
        ...base,
        rotationDeg: base.rotationDeg + (rand() - 0.5) * 0.6,
        blur: Math.max(0, base.blur + (rand() - 0.5) * 0.4),
        offsetX: base.offsetX + (rand() - 0.5) * 0.004,
      };
      const { frame } = await synthesize(img, 1080, 1920, guide, p, rand);
      const a = analyzeFrame(frame, { guide, infoLine });
      t += 150;
      if (!a.accepted || !a.quad) continue;
      const read = readCard(
        model,
        frame,
        a.alternatives.length ? a.alternatives : [a.quad],
        infoLine,
      );
      const state = fusion.push({ info: read.info }, t);
      if (state.status === 'confirmed' && state.best) {
        outcome =
          state.best.identity.key === cardKey(card.code, card.number)
            ? 'correct'
            : `WRONG ${state.best.identity.key}`;
        framesToConfirm += f + 1;
        break;
      }
    }
    if (outcome === 'correct') correct++;
    else if (outcome === 'none') none++;
    else wrong++;
    if (outcome !== 'correct' || !args.quiet)
      console.log(
        `${outcome.padEnd(16)} ${card.label.padEnd(13)} bg=${base.background} blur=${base.blur.toFixed(1)} glare=${base.glare}`,
      );
  }
  const n = picks.length;
  console.log(
    `\n=== fusion (${k} frames per card, catalog ${index ? 'on' : 'OFF (simulates a brand-new set)'}) ===`,
  );
  console.log(
    `captured correctly  ${((100 * correct) / n).toFixed(1)}%   avg frames to confirm ${(framesToConfirm / Math.max(1, correct + wrong)).toFixed(1)}`,
  );
  console.log(`captured WRONG      ${((100 * wrong) / n).toFixed(1)}%`);
  console.log(`not captured        ${((100 * none) / n).toFixed(1)}%`);
}

async function main(): Promise<void> {
  const catalog = JSON.parse(
    await readFile(join(ROOT, 'public', 'catalog.json'), 'utf8'),
  ) as Catalog;
  const index = new CatalogIndex(catalog);
  const detectOnly = args['detect-only'];
  let model: LineModel | null = null;
  let manifest: LineModelManifest | null = null;
  if (!detectOnly) {
    manifest = JSON.parse(
      await readFile(join(ROOT, 'public', 'models', 'info-line.json'), 'utf8'),
    ) as LineModelManifest;
    const weights = await readFile(join(ROOT, 'public', 'models', manifest.weights));
    model = LineModel.fromBuffers(
      manifest,
      weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength),
    );
  }
  const infoLine = model
    ? { region: model.region, ...model.inputSize }
    : { region: INFO_LINE, ...INFO_LINE_INPUT };

  const scans = JSON.parse(
    await readFile(join(CACHE, 'scans', 'manifest.json'), 'utf8'),
  ) as ScanLabel[];
  const wanted = args.sets?.split(',');
  const pool = scans.filter((s) => !wanted || wanted.includes(s.code));
  const rand = mulberry32(parseInt(args.seed!, 10));
  const n = parseInt(args.n!, 10);
  const picks = Array.from({ length: n }, () => pool[Math.floor(rand() * pool.length)]);

  if (parseInt(args.fusion!, 10) > 0 && model) {
    await fusionEval(model, args['no-catalog'] ? null : index, picks, parseInt(args.fusion!, 10));
    return;
  }

  const outDir = join(CACHE, 'eval', new Date().toISOString().replace(/[:.]/g, '-'));
  if (args['save-failures']) await mkdir(outDir, { recursive: true });
  const engine = args.names ? new TesseractEngine() : null;
  await engine?.warmUp();

  const frameW = 1080,
    frameH = 1920;
  const guide = defaultGuide(frameW, frameH);
  const s = {
    detected: 0,
    cornerOk: 0,
    exact: 0,
    confident: 0,
    confidentCorrect: 0,
    resolved: 0,
    names: 0,
    nameReads: 0,
    detectMs: 0,
    modelMs: 0,
    nameMs: 0,
  };
  const cornerErrs: number[] = [];
  const byBg = new Map<string, { n: number; ok: number }>();

  for (let i = 0; i < picks.length; i++) {
    const card = picks[i];
    const truthKey = cardKey(card.code, card.number);
    const img: RGBAImage = await loadRGBA(join(CACHE, 'scans', card.file));
    let crops: FrameCrops;
    let info: InfoRead | null = null;
    let detail = '';
    let frame: RGBAImage | null = null;
    if (args.clean) {
      const H = cardToImage([
        { x: 0, y: 0 },
        { x: img.width, y: 0 },
        { x: img.width, y: img.height },
        { x: 0, y: img.height },
      ]);
      crops = {
        infoLine: warpCardRegion(img, H, infoLine.region, infoLine.width, infoLine.height, 2),
        name: cropRegion(img, H, NAME_REGION, NAME_CARD_WIDTH),
      };
      s.detected++;
      s.cornerOk++;
    } else {
      const params = randomParams(rand, args.difficulty as 'easy' | 'normal' | 'hard');
      const synth = await synthesize(img, frameW, frameH, guide, params, rand);
      frame = synth.frame;
      const analysis = analyzeFrame(synth.frame, { guide, infoLine });
      s.detectMs += analysis.timings.detect + analysis.timings.warp;
      const err = analysis.quad
        ? (cornerError(analysis.quad, synth.quad) / guide.w) * 100
        : Infinity;
      detail = `bg=${params.background} rot=${params.rotationDeg.toFixed(1)} tilt=${params.tiltX.toFixed(0)},${params.tiltY.toFixed(0)} scale=${params.scale.toFixed(2)} blur=${params.blur.toFixed(1)} glare=${params.glare} score=${analysis.scores?.total.toFixed(2) ?? '-'} cornerErr=${err.toFixed(1)}%`;
      if (!analysis.accepted || !analysis.crops) {
        console.log(`MISS ${truthKey.padEnd(9)} not detected  ${detail}`);
        if (args['save-failures'])
          await saveRGBA(synth.frame, join(outDir, `${i}-${truthKey}-frame.jpg`));
        continue;
      }
      s.detected++;
      cornerErrs.push(err);
      const bg = byBg.get(params.background) ?? { n: 0, ok: 0 };
      bg.n++;
      if (err < 1.5) bg.ok++;
      byBg.set(params.background, bg);
      if (err < 1.5) s.cornerOk++;
      crops = analysis.crops;
      if (model && analysis.quad) {
        const read = readCard(
          model,
          synth.frame,
          analysis.alternatives.length ? analysis.alternatives : [analysis.quad],
          infoLine,
        );
        info = read.info;
        crops = read.crops;
        if (read.variant !== 'normal') detail += ` via=${read.variant}`;
      }
      if (detectOnly) {
        if (!args.quiet || err >= 1.5)
          console.log(`${err < 1.5 ? 'ok  ' : 'POOR'} ${truthKey.padEnd(9)} ${detail}`);
        if (err >= 1.5 && args['save-failures'])
          await saveRGBA(synth.frame, join(outDir, `${i}-${truthKey}-frame.jpg`));
        continue;
      }
    }
    if (!model) continue;

    info ??= readInfoLine(model, crops.infoLine);
    s.modelMs += info.ms;
    const exact = info.text === card.label;
    if (exact) s.exact++;
    const confident = !!info.parsed && info.confidence >= 0.5;
    if (confident) {
      s.confident++;
      if (exact) s.confidentCorrect++;
    }
    let resolvedKey = '-';
    if (info.parsed) {
      const r = index.resolve(info.parsed.setCode, info.parsed.number, info.parsed.total, null);
      resolvedKey = cardKey(r.setCode, r.number);
      if (resolvedKey === truthKey) s.resolved++;
    }
    let nameInfo = '';
    if (engine) {
      const name = await readName(engine, crops.name);
      s.nameMs += name.ms;
      if (name.value) s.nameReads++;
      const sim = name.value ? nameSimilarity(name.value, card.name) : 0;
      if (sim >= 0.6) s.names++;
      nameInfo = ` name=${JSON.stringify(name.value)}${sim >= 0.6 ? '' : '✗'}`;
    }
    const ok = resolvedKey === truthKey;
    if (!ok || !args.quiet) {
      console.log(
        `${ok ? 'ok  ' : 'FAIL'} ${card.label.padEnd(13)} read=${JSON.stringify(info.text).padEnd(15)} conf=${info.confidence.toFixed(2)}${nameInfo}  ${detail}`,
      );
    }
    if (!ok && args['save-failures']) {
      const base = join(outDir, `${i}-${truthKey}`);
      if (frame) await saveRGBA(frame, `${base}-frame.jpg`);
      await saveRGBA(crops.infoLine, `${base}-info.png`);
      await saveGray(
        {
          width: infoLine.width,
          height: infoLine.height,
          data: model.prepare(crops.infoLine).map((v) => (v + 0.5) * 255),
        },
        `${base}-info-input.png`,
      );
    }
  }
  await engine?.terminate();

  const pct = (k: number, d = n) => `${((100 * k) / Math.max(1, d)).toFixed(1)}%`.padStart(6);
  cornerErrs.sort((a, b) => a - b);
  console.log('\n=== summary ===');
  console.log(
    `model             ${manifest ? `${manifest.name} (${manifest.createdAt}) ${manifest.note ?? ''}` : 'none (detection only)'}`,
  );
  console.log(
    `frames            ${n} (${args.clean ? 'clean scans' : `synthetic, ${args.difficulty}`}${wanted ? `, sets ${wanted.join(',')}` : ''})`,
  );
  const q = (p: number) =>
    cornerErrs[Math.min(cornerErrs.length - 1, Math.floor(cornerErrs.length * p))]?.toFixed(2) ??
    '-';
  console.log(
    `detected          ${pct(s.detected)}   corner err <1.5% of card width: ${pct(s.cornerOk)}   p50 ${q(0.5)}% p90 ${q(0.9)}% p95 ${q(0.95)}%`,
  );
  console.log(
    `  corner ok by bg  ${[...byBg.entries()].map(([k, v]) => `${k} ${v.ok}/${v.n}`).join('  ')}`,
  );
  if (detectOnly) return;
  console.log(`info line exact   ${pct(s.exact)}   (of detected: ${pct(s.exact, s.detected)})`);
  console.log(
    `confident reads   ${pct(s.confident)}   precision when confident: ${pct(s.confidentCorrect, s.confident)}`,
  );
  console.log(`resolved key      ${pct(s.resolved)}   <- single-frame identification`);
  if (engine)
    console.log(`name              ${pct(s.names)}   (read something: ${pct(s.nameReads)})`);
  console.log(
    `timing            detect+warp ${(s.detectMs / Math.max(1, s.detected)).toFixed(0)} ms, model ${(s.modelMs / Math.max(1, s.detected)).toFixed(0)} ms${engine ? `, name OCR ${(s.nameMs / Math.max(1, s.detected)).toFixed(0)} ms` : ''}`,
  );
  if (args['save-failures']) console.log(`failure images in ${outDir}`);
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
