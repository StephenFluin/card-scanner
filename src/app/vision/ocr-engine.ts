import type { GrayImage } from './image';
import type { OcrWord } from './parse';

/** Fields read with general-purpose OCR (the info line uses our own model). */
export type OcrField = 'name';

export interface OcrOutput {
  text: string;
  /** 0..100 */
  confidence: number;
  words: OcrWord[];
  ms: number;
}

/**
 * Anything that can read text from a prepared (dark text on white) image.
 */
export interface OcrEngine {
  readonly name: string;
  recognize(field: OcrField, image: GrayImage): Promise<OcrOutput>;
  terminate(): Promise<void>;
}
