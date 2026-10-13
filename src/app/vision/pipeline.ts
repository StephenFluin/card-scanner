import { CardRegion, INFO_LINE, INFO_LINE_INPUT, NAME_CARD_WIDTH, NAME_REGION } from './card-layout';
import { CARD_ASPECT, DetectResult, DetectScores, detectCard, EdgeLine, EdgePoint, Side } from './detect';
import { cardToImage, Mat3, Quad, rotateQuad180, warpCardRegion } from './homography';
import { crop, Rect, resizeArea, RGBAImage } from './image';
import { sharpness } from './ocr-prep';

/**
 * Synchronous per-frame stage: find the card, rectify it and cut out the
 * regions to OCR. Pure and DOM-free so it can run in a Web Worker.
 */

export interface AnalyzeOptions {
  /** Expected card position in frame pixels. */
  guide: Rect;
  /** Width (px) the guide is scaled to for edge detection. */
  detectWidth?: number;
  /** Minimum detection score to bother rectifying. */
  minScore?: number;
  /** Treat the card as upside down. */
  flip?: boolean;
  /** Also produce a small rectified preview of the whole card. */
  preview?: boolean;
  /** Info-line model input (region + pixel size); defaults to the built-in layout. */
  infoLine?: InfoLineSpec;
  debug?: boolean;
}

export interface FrameCrops {
  /** Exactly the info-line model's input size. */
  infoLine: RGBAImage;
  name: RGBAImage;
}

export interface FrameAnalysis {
  /** Card corners in frame pixels (reading order), or null. */
  quad: Quad | null;
  scores: DetectScores | null;
  /** True when the detection is good enough to OCR. */
  accepted: boolean;
  crops: FrameCrops | null;
  preview: RGBAImage | null;
  /** Laplacian variance of the info-line crop (focus indicator). */
  sharpness: number;
  timings: { detect: number; warp: number };
  /** Detection internals in frame pixels, for overlays. */
  debug?: {
    roi: Rect;
    scale: number;
    points: Record<Side, EdgePoint[]>;
    candidates: Record<Side, EdgeLine[]>;
  };
}

export const DEFAULT_MIN_SCORE = 0.62;

export function cropRegion(frame: RGBAImage, H: Mat3, region: CardRegion, cardWidthPx: number): RGBAImage {
  const cardHeightPx = cardWidthPx / CARD_ASPECT;
  const w = Math.max(8, Math.round(region.w * cardWidthPx));
  const h = Math.max(8, Math.round(region.h * cardHeightPx));
  return warpCardRegion(frame, H, region, w, h, 2);
}

export type InfoLineSpec = { region: CardRegion; width: number; height: number };

/** Rectifies the regions we read from a card with the given corners. */
export function cropsForQuad(frame: RGBAImage, quad: Quad, infoLine?: InfoLineSpec): FrameCrops {
  const H = cardToImage(quad);
  const info = infoLine ?? { region: INFO_LINE, ...INFO_LINE_INPUT };
  return {
    // 2x2 supersampling matches the training-time rendering (2x warp + area downscale).
    infoLine: warpCardRegion(frame, H, info.region, info.width, info.height, 2),
    name: cropRegion(frame, H, NAME_REGION, NAME_CARD_WIDTH),
  };
}

/** Region of interest around the guide where the card edges can be. */
export function detectionRoi(frame: { width: number; height: number }, guide: Rect): Rect {
  const m = guide.w * 0.28;
  const x = Math.max(0, Math.floor(guide.x - m));
  const y = Math.max(0, Math.floor(guide.y - m));
  return {
    x,
    y,
    w: Math.min(frame.width, Math.ceil(guide.x + guide.w + m)) - x,
    h: Math.min(frame.height, Math.ceil(guide.y + guide.h + m)) - y,
  };
}

export function analyzeFrame(frame: RGBAImage, opts: AnalyzeOptions): FrameAnalysis {
  const t0 = performance.now();
  const detectWidth = opts.detectWidth ?? 300;
  const roi = detectionRoi(frame, opts.guide);
  const scale = detectWidth / opts.guide.w;
  const small = resizeArea(crop(frame, roi), Math.round(roi.w * scale), Math.round(roi.h * scale));
  const guideSmall: Rect = {
    x: (opts.guide.x - roi.x) * scale,
    y: (opts.guide.y - roi.y) * scale,
    w: opts.guide.w * scale,
    h: opts.guide.h * scale,
  };
  const det: DetectResult = detectCard(small, { guide: guideSmall, debug: opts.debug });
  const toFrame = (p: { x: number; y: number }) => ({ x: p.x / scale + roi.x, y: p.y / scale + roi.y });
  let quad: Quad | null = det.quad ? (det.quad.map(toFrame) as unknown as Quad) : null;
  const t1 = performance.now();

  const accepted = !!quad && !!det.scores && det.scores.total >= (opts.minScore ?? DEFAULT_MIN_SCORE);
  let crops: FrameCrops | null = null;
  let preview: RGBAImage | null = null;
  let focus = 0;
  if (quad && accepted) {
    if (opts.flip) quad = rotateQuad180(quad);
    crops = cropsForQuad(frame, quad, opts.infoLine);
    focus = sharpness(crops.infoLine);
  }
  if (quad && opts.preview) {
    const H = cardToImage(opts.flip && !accepted ? rotateQuad180(quad) : quad);
    preview = warpCardRegion(frame, H, { x: 0, y: 0, w: 1, h: 1 }, 252, Math.round(252 / CARD_ASPECT), 2);
  }
  const t2 = performance.now();

  let debug: FrameAnalysis['debug'];
  if (opts.debug && det.debug) {
    const mapPts = (pts: EdgePoint[], side: Side): EdgePoint[] =>
      pts.map((p) => {
        const vertical = side === 'left' || side === 'right';
        return vertical
          ? { t: p.t / scale + roi.y, s: p.s / scale + roi.x, m: p.m }
          : { t: p.t / scale + roi.x, s: p.s / scale + roi.y, m: p.m };
      });
    const mapLine = (l: EdgeLine): EdgeLine => {
      const vertical = l.side === 'left' || l.side === 'right';
      // s = a t + b in small coords -> frame coords.
      const [tOff, sOff] = vertical ? [roi.y, roi.x] : [roi.x, roi.y];
      return { ...l, b: l.b / scale + sOff - l.a * tOff };
    };
    const sides: Side[] = ['left', 'right', 'top', 'bottom'];
    debug = {
      roi,
      scale,
      points: Object.fromEntries(sides.map((s) => [s, mapPts(det.debug!.points[s], s)])) as Record<Side, EdgePoint[]>,
      candidates: Object.fromEntries(sides.map((s) => [s, det.debug!.candidates[s].map(mapLine)])) as Record<Side, EdgeLine[]>,
    };
  }

  return {
    quad,
    scores: det.scores,
    accepted,
    crops,
    preview,
    sharpness: focus,
    timings: { detect: t1 - t0, warp: t2 - t1 },
    debug,
  };
}

/**
 * Default guide: a portrait card-shaped rectangle centred in the frame,
 * filling most of the short dimension.
 */
export function defaultGuide(width: number, height: number, fill = 0.78): Rect {
  let h = height * fill;
  let w = h * CARD_ASPECT;
  if (w > width * fill) {
    w = width * fill;
    h = w / CARD_ASPECT;
  }
  return { x: (width - w) / 2, y: (height - h) / 2, w, h };
}
