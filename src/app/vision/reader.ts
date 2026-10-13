import type { GrayImage, RGBAImage } from './image';
import { LineModel, parseInfoLine, ParsedInfoLine } from './line-model';
import type { OcrEngine } from './ocr-engine';
import { OcrPrepOptions, prepareForOcr } from './ocr-prep';
import { parseName } from './parse';

/**
 * Reading a rectified card:
 *  - the info line ("SVI 001/198") with our own trained model (fast, robust);
 *  - optionally the name with Tesseract (open vocabulary, slower, only used
 *    for display / sanity checks, never required for identification).
 */

export interface InfoRead {
  text: string;
  /** Lowest per-character probability, 0..1. */
  confidence: number;
  charConfidences: number[];
  parsed: ParsedInfoLine | null;
  ms: number;
}

export interface NameAttempt {
  variant: string;
  text: string;
  value: string | null;
  confidence: number;
  ms: number;
}

export interface NameRead {
  value: string | null;
  confidence: number;
  attempts: NameAttempt[];
  /** The image of the chosen attempt. */
  prepared: GrayImage;
  ms: number;
}

export function readInfoLine(model: LineModel, crop: RGBAImage): InfoRead {
  const result = model.run(model.prepare(crop));
  return {
    text: result.text,
    confidence: result.confidence,
    charConfidences: result.charConfidences,
    parsed: parseInfoLine(result.text),
    ms: result.ms,
  };
}

const NAME_VARIANTS: { label: string; prep: OcrPrepOptions }[] = [
  { label: 'dark-text', prep: { targetHeight: 140, polarity: 'dark-text', binarize: false, pad: 16 } },
  { label: 'light-text', prep: { targetHeight: 140, polarity: 'light-text', binarize: false, pad: 16 } },
];

/**
 * Names can be dark-on-light or light-on-dark depending on the card type,
 * so both polarities are tried and the more confident read wins.
 */
export async function readName(engine: OcrEngine, crop: RGBAImage): Promise<NameRead> {
  const start = performance.now();
  const attempts: NameAttempt[] = [];
  let best: { attempt: NameAttempt; image: GrayImage } | null = null;
  let first: GrayImage | null = null;
  for (const v of NAME_VARIANTS) {
    const prepared = prepareForOcr(crop, v.prep);
    first ??= prepared.image;
    const out = await engine.recognize('name', prepared.image);
    const value = parseName(out.words);
    const parts = value?.split(' ') ?? [];
    const used = out.words.filter((w) => parts.some((p) => p.length > 0 && w.text.includes(p)));
    const confidence = used.length ? used.reduce((s, w) => s + w.confidence, 0) / used.length : 0;
    const attempt: NameAttempt = { variant: v.label, text: out.text, value, confidence, ms: out.ms };
    attempts.push(attempt);
    if (value && (!best || confidence > best.attempt.confidence)) best = { attempt, image: prepared.image };
    if (value && confidence >= 85) break;
  }
  return {
    value: best?.attempt.value ?? null,
    confidence: best?.attempt.confidence ?? 0,
    attempts,
    prepared: best?.image ?? first!,
    ms: performance.now() - start,
  };
}
