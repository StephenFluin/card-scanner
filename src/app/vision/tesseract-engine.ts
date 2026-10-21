import { createWorker, OEM, PSM, type Worker, type WorkerOptions } from 'tesseract.js';
import type { GrayImage } from './image';
import type { OcrEngine, OcrField, OcrOutput } from './ocr-engine';
import { encodePgm } from './ocr-prep';
import type { OcrWord } from './parse';

const FIELD_PARAMS: Record<OcrField, { psm: PSM; whitelist: string }> = {
  name: { psm: PSM.SINGLE_BLOCK, whitelist: '' },
};

/**
 * Tesseract (LSTM) OCR, one worker per field so each keeps its own page
 * segmentation mode and character whitelist.
 */
export class TesseractEngine implements OcrEngine {
  readonly name = 'tesseract';
  private readonly workers = new Map<OcrField, Promise<Worker>>();

  constructor(private readonly options: Partial<WorkerOptions> = {}) {}

  private worker(field: OcrField): Promise<Worker> {
    let w = this.workers.get(field);
    if (!w) {
      w = (async () => {
        const worker = await createWorker('eng', OEM.LSTM_ONLY, this.options);
        const p = FIELD_PARAMS[field];
        await worker.setParameters({
          tessedit_pageseg_mode: p.psm,
          tessedit_char_whitelist: p.whitelist,
          // Text is never inverted after our preprocessing; skip the extra pass.
          tessedit_do_invert: '0',
        });
        return worker;
      })();
      this.workers.set(field, w);
    }
    return w;
  }

  /** Starts all workers ahead of time so the first scan isn't slow. */
  async warmUp(): Promise<void> {
    await this.worker('name');
  }

  async recognize(field: OcrField, image: GrayImage): Promise<OcrOutput> {
    const worker = await this.worker(field);
    const start = performance.now();
    const pgm = encodePgm(image);
    const input = typeof Buffer !== 'undefined' ? Buffer.from(pgm) : new Blob([pgm as BlobPart]);
    const { data } = await worker.recognize(input, {}, { text: true, blocks: true });
    const words: OcrWord[] = [];
    for (const block of data.blocks ?? []) {
      for (const para of block.paragraphs) {
        for (const line of para.lines) {
          for (const w of line.words) {
            words.push({
              text: w.text,
              confidence: w.confidence,
              height: w.bbox.y1 - w.bbox.y0,
              y: (w.bbox.y0 + w.bbox.y1) / 2,
              x: w.bbox.x0,
            });
          }
        }
      }
    }
    return {
      text: data.text.trim(),
      confidence: data.confidence,
      words,
      ms: performance.now() - start,
    };
  }

  async terminate(): Promise<void> {
    const all = [...this.workers.values()];
    this.workers.clear();
    await Promise.all(all.map(async (w) => (await w).terminate()));
  }
}
