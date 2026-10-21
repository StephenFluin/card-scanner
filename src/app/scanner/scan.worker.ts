/// <reference lib="webworker" />
/**
 * Runs the per-frame vision pipeline off the main thread:
 * find card edges -> rectify -> read the info line with our model.
 */
import { CARD_ASPECT } from '../vision/detect';
import { cardToImage, warpCardRegion } from '../vision/homography';
import type { RGBAImage } from '../vision/image';
import { LineModel } from '../vision/line-model';
import { analyzeFrame, readCard, ReadVariant } from '../vision/pipeline';
import type { InfoRead } from '../vision/reader';
import {
  CARD_IMAGE_WIDTH,
  FrameRequest,
  FrameResult,
  WorkerRequest,
  WorkerResponse,
} from './scan-protocol';

let model: LineModel | null = null;
/** Running average of one model run (ms), to size the per-frame hypothesis budget. */
let modelMs = 40;
const FRAME_BUDGET_MS = 250;
/** Consecutive frames of the same scene usually need the same hypothesis. */
let lastWinner: ReadVariant | null = null;
let canvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []): void {
  postMessage(msg, transfer);
}

function grab(bitmap: ImageBitmap): RGBAImage {
  if (!canvas || canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }
  ctx!.drawImage(bitmap, 0, 0);
  const img = ctx!.getImageData(0, 0, bitmap.width, bitmap.height);
  return { width: img.width, height: img.height, data: img.data };
}

function handleFrame(req: FrameRequest): void {
  const t0 = performance.now();
  const frame = grab(req.bitmap);
  req.bitmap.close();
  const t1 = performance.now();

  const infoLine = model ? { region: model.region, ...model.inputSize } : undefined;
  const analysis = analyzeFrame(frame, {
    guide: req.guide,
    debug: req.debug,
    preview: req.debug,
    infoLine,
  });
  let quad = analysis.quad;
  let crops = analysis.crops;
  let info: InfoRead | null = null;
  let variant: ReadVariant | null = null;
  let tried: { variant: ReadVariant; text: string; confidence: number }[] = [];
  const t2 = performance.now();
  if (model && quad && crops && analysis.accepted) {
    const maxReads = Math.min(8, Math.max(2, Math.floor(FRAME_BUDGET_MS / modelMs)));
    const read = readCard(
      model,
      frame,
      analysis.alternatives.length ? analysis.alternatives : [quad],
      infoLine,
      maxReads,
      lastWinner,
    );
    ({ info, quad, crops, variant, tried } = read);
    lastWinner = read.info.parsed ? read.variant : null;
    const perRead = (performance.now() - t2) / Math.max(1, tried.length);
    modelMs = modelMs * 0.8 + perRead * 0.2;
  }
  const t3 = performance.now();

  const transfer: Transferable[] = [];
  const keep = (img: RGBAImage | null | undefined): RGBAImage | null => {
    if (!img) return null;
    transfer.push(img.data.buffer);
    return img;
  };
  const result: FrameResult = {
    type: 'result',
    id: req.id,
    width: frame.width,
    height: frame.height,
    guide: req.guide,
    quad,
    scores: analysis.scores,
    accepted: analysis.accepted,
    variant,
    tried,
    sharpness: analysis.sharpness,
    info,
    nameCrop: req.wantNameCrop ? keep(crops?.name) : null,
    cardImage:
      req.wantCardImage && quad && info?.parsed
        ? keep(
            warpCardRegion(
              frame,
              cardToImage(quad),
              { x: 0, y: 0, w: 1, h: 1 },
              CARD_IMAGE_WIDTH,
              Math.round(CARD_IMAGE_WIDTH / CARD_ASPECT),
              2,
            ),
          )
        : null,
    timings: {
      capture: t1 - t0,
      detect: analysis.timings.detect,
      warp: analysis.timings.warp,
      model: t3 - t2,
      total: performance.now() - t0,
    },
  };
  if (req.debug && analysis.debug) {
    result.debug = {
      roi: analysis.debug.roi,
      points: analysis.debug.points,
      candidates: analysis.debug.candidates,
      infoCrop: keep(crops?.infoLine),
      preview: keep(analysis.preview),
    };
  }
  if (req.keepFrame) result.frame = keep(frame) ?? undefined;
  post(result, transfer);
}

addEventListener('message', async ({ data }: MessageEvent<WorkerRequest>) => {
  try {
    if (data.type === 'init') {
      model = await LineModel.load(data.modelUrl);
      const m = model.manifest;
      post({
        type: 'ready',
        model: { name: m.name, createdAt: m.createdAt, note: m.note, params: m.params },
      });
    } else if (data.type === 'frame') {
      handleFrame(data);
    }
  } catch (e) {
    post({
      type: 'error',
      id: data.type === 'frame' ? data.id : undefined,
      message: e instanceof Error ? `${e.message}\n${e.stack}` : String(e),
    });
  }
});
