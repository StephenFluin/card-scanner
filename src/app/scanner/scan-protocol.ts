import type { DetectScores, EdgeLine, EdgePoint, Side } from '../vision/detect';
import type { Quad } from '../vision/homography';
import type { Rect, RGBAImage } from '../vision/image';
import type { ReadVariant } from '../vision/pipeline';
import type { InfoRead } from '../vision/reader';

/** Messages between the scan controller (main thread) and scan.worker.ts. */

/** Width of contributed card images; the info line is then ~model resolution. */
export const CARD_IMAGE_WIDTH = 720;

export interface InitRequest {
  type: 'init';
  /** Absolute URL of the info-line model manifest. */
  modelUrl: string;
}

export interface FrameRequest {
  type: 'frame';
  id: number;
  bitmap: ImageBitmap;
  /** Where the user was asked to hold the card, in bitmap pixels. */
  guide: Rect;
  /** Return the rectified name crop (for Tesseract on the main thread). */
  wantNameCrop: boolean;
  /** Return a straightened image of the whole card (contributed for training). */
  wantCardImage: boolean;
  /** Return detection internals and intermediate images. */
  debug: boolean;
  /** Also return the full frame pixels (for debug snapshots). */
  keepFrame: boolean;
}

export type WorkerRequest = InitRequest | FrameRequest;

export interface ModelInfo {
  name: string;
  createdAt: string;
  note?: string;
  params: number;
}

export interface FrameDebug {
  roi: Rect;
  points: Record<Side, EdgePoint[]>;
  candidates: Record<Side, EdgeLine[]>;
  /** Rectified info line, exactly what the model saw (before normalization). */
  infoCrop: RGBAImage | null;
  /** Small rectified preview of the whole card. */
  preview: RGBAImage | null;
}

export interface FrameResult {
  type: 'result';
  id: number;
  width: number;
  height: number;
  guide: Rect;
  /** Card corners in reading order (already rotated if the card is upside down). */
  quad: Quad | null;
  scores: DetectScores | null;
  accepted: boolean;
  /** Which hypothesis produced the read: upright/upside down, detected or border-expanded quad. */
  variant: ReadVariant | null;
  /** Every hypothesis tried this frame. */
  tried: { variant: ReadVariant; text: string; confidence: number }[];
  sharpness: number;
  info: InfoRead | null;
  nameCrop: RGBAImage | null;
  /** Straightened card (CARD_IMAGE_WIDTH wide), when requested and a card was read. */
  cardImage: RGBAImage | null;
  timings: { capture: number; detect: number; warp: number; model: number; total: number };
  debug?: FrameDebug;
  frame?: RGBAImage;
}

export type WorkerResponse =
  | { type: 'ready'; model: ModelInfo }
  | { type: 'error'; id?: number; message: string }
  | FrameResult;
