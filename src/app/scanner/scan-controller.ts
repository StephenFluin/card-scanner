import { computed, inject, Injectable, OnDestroy, signal } from '@angular/core';
import { CatalogService } from '../services/catalog.service';
import { CollectionService } from '../services/collection.service';
import { ContributionService, Observation } from '../services/contribution.service';
import { Logger } from '../services/logger.service';
import type { CardIdentity } from '../vision/catalog';
import { DEFAULT_FUSION, FusionState, ReadFusion } from '../vision/fusion';
import type { Rect, RGBAImage } from '../vision/image';
import { defaultGuide } from '../vision/pipeline';
import type { NameRead } from '../vision/reader';
import type { FrameSource } from './frame-source';
import type { FrameRequest, FrameResult, ModelInfo, WorkerResponse } from './scan-protocol';

export type ScanStatus =
  | 'loading' // model / worker starting
  | 'waiting' // no frames yet
  | 'searching' // no card in the guide
  | 'reading' // card found, reading the info line
  | 'captured' // just captured a card
  | 'holding' // same card still in view after capture
  | 'paused'
  | 'error';

export interface Capture {
  id: number;
  identity: CardIdentity;
  setName: string | null;
  imageUrl: string | null;
  known: boolean;
  at: number;
  reason: string;
}

/** Compact per-frame record kept for the debug panel and bundles. */
export interface FrameSummary {
  id: number;
  t: number;
  accepted: boolean;
  score: number | null;
  read: string | null;
  confidence: number | null;
  variant: string | null;
  vote: string;
  status: FusionState['status'] | 'none';
  timings: FrameResult['timings'];
}

export interface Snapshot {
  at: string;
  result: FrameResult;
  fusion: FusionState | null;
  name: NameRead | null;
}

export interface ScanSettings {
  /** Guide size as a fraction of the frame's short side. */
  guideFill: number;
  /** Run Tesseract on the name for every candidate, not just unknown cards. */
  alwaysReadNames: boolean;
  /** Minimum ms between frames sent to the worker. */
  minFrameInterval: number;
}

const HISTORY = 60;
/** Frames without a card before the scanner re-arms for the same card. */
const REARM_FRAMES = 4;

/**
 * Drives scanning: pulls frames from a source, runs them through the vision
 * worker, fuses reads across frames, and captures confirmed cards into the
 * collection. One instance per scanner page.
 */
@Injectable()
export class ScanController implements OnDestroy {
  private readonly logger = inject(Logger);
  private readonly catalog = inject(CatalogService);
  private readonly collection = inject(CollectionService);
  private readonly contribution = inject(ContributionService);

  readonly status = signal<ScanStatus>('loading');
  readonly statusDetail = signal<string>('Loading recognition model…');
  readonly model = signal<ModelInfo | null>(null);
  readonly result = signal<FrameResult | null>(null);
  readonly fusion = signal<FusionState | null>(null);
  readonly captures = signal<Capture[]>([]);
  readonly lastCapture = computed(() => this.captures()[0] ?? null);
  readonly history = signal<FrameSummary[]>([]);
  readonly fps = signal(0);
  readonly debug = signal(false);
  readonly paused = signal(false);
  readonly lastName = signal<NameRead | null>(null);
  readonly snapshot = signal<Snapshot | null>(null);
  readonly settings = signal<ScanSettings>({ guideFill: 0.8, alwaysReadNames: false, minFrameInterval: 70 });
  readonly sourceLabel = signal<string>('');

  private worker: Worker | null = null;
  private readonly fuser = new ReadFusion(null, DEFAULT_FUSION);
  private source: FrameSource | null = null;
  private running = false;
  private nextId = 1;
  private pending = new Map<number, (r: FrameResult | null) => void>();
  private armed = true;
  private lastCapturedKey: string | null = null;
  private framesWithoutCard = 0;
  private lastFrameAt = 0;
  private nameInFlight = false;
  private pendingName: string | null = null;
  private namedKeys = new Set<string>();
  private wantSnapshot = false;
  private tesseract: Promise<import('../vision/tesseract-engine').TesseractEngine> | null = null;
  /** A capture waiting for a card image (taken from the next frames while the card is in view). */
  private pendingContribution: { observation: Observation; framesLeft: number } | null = null;
  private captureSeq = 0;

  constructor() {
    this.startWorker();
  }

  ngOnDestroy(): void {
    this.flushContribution();
    this.stop();
    this.worker?.terminate();
    void this.tesseract?.then((t) => t.terminate());
  }

  /** The guide rectangle (frame pixels) for a frame of the given size. */
  guideFor(width: number, height: number): Rect {
    return defaultGuide(width, height, this.settings().guideFill);
  }

  setSource(source: FrameSource | null): void {
    this.source?.dispose();
    this.source = source;
    this.sourceLabel.set(source ? `${source.kind}: ${source.label}` : '');
    this.fuser.reset();
    this.armed = true;
    this.lastCapturedKey = null;
    this.logger.info('app', source ? `frame source: ${source.kind} (${source.label})` : 'frame source cleared');
    if (source && !this.running) void this.loop();
  }

  stop(): void {
    this.running = false;
  }

  togglePause(): void {
    this.paused.update((p) => !p);
    this.status.set(this.paused() ? 'paused' : 'searching');
    this.logger.info('app', this.paused() ? 'paused' : 'resumed');
  }

  requestSnapshot(): void {
    this.wantSnapshot = true;
  }

  /** Removes the most recent capture from the collection (wrong card, double scan...). */
  async undo(capture: Capture): Promise<void> {
    await this.collection.decrement(capture.identity.key);
    this.captures.update((list) => list.filter((c) => c.id !== capture.id));
    this.logger.info('capture', `undo ${capture.identity.key}`);
  }

  /**
   * The capture was the wrong card: remove it and record the misread, which
   * counts against it when the catalog is learned from scans.
   */
  async wrongCard(capture: Capture): Promise<void> {
    await this.undo(capture);
    await this.contribution.reject(capture.identity);
    this.logger.warn('capture', `user reported ${capture.identity.key} as the wrong card`);
  }

  /** Adds another copy of a captured card without rescanning it. */
  async addAnother(capture: Capture): Promise<void> {
    await this.collection.add(capture.identity);
    this.logger.info('capture', `+1 ${capture.identity.key}`);
  }

  private startWorker(): void {
    this.worker = new Worker(new URL('./scan.worker', import.meta.url), { type: 'module' });
    this.worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => this.onWorkerMessage(data);
    this.worker.onerror = (e) => {
      this.logger.error('worker', `worker crashed: ${e.message}`);
      this.status.set('error');
      this.statusDetail.set('The recognition worker crashed. See the debug log.');
    };
    const modelUrl = new URL('models/info-line.json', document.baseURI).toString();
    this.worker.postMessage({ type: 'init', modelUrl });
    this.logger.info('worker', `starting, model ${modelUrl}`);
  }

  private onWorkerMessage(msg: WorkerResponse): void {
    if (msg.type === 'ready') {
      this.model.set(msg.model);
      this.status.set('waiting');
      this.statusDetail.set('Starting camera…');
      this.logger.info('worker', `model ready: ${msg.model.name} (${msg.model.params} params, ${msg.model.createdAt})`);
      if (this.source && !this.running) void this.loop();
    } else if (msg.type === 'error') {
      this.logger.error('worker', msg.message);
      if (msg.id === undefined) {
        this.status.set('error');
        this.statusDetail.set(`Could not load the recognition model: ${msg.message.split('\n')[0]}`);
      } else {
        this.pending.get(msg.id)?.(null);
        this.pending.delete(msg.id);
      }
    } else {
      this.pending.get(msg.id)?.(msg);
      this.pending.delete(msg.id);
    }
  }

  private send(bitmap: ImageBitmap, guide: Rect, wantNameCrop: boolean, keepFrame: boolean): Promise<FrameResult | null> {
    const id = this.nextId++;
    const wantCardImage = !!this.pendingContribution;
    const req: FrameRequest = { type: 'frame', id, bitmap, guide, wantNameCrop, wantCardImage, debug: this.debug() || keepFrame, keepFrame };
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.worker!.postMessage(req, [bitmap]);
    });
  }

  private async loop(): Promise<void> {
    if (this.running || !this.model()) return;
    this.running = true;
    this.logger.info('app', 'scan loop started');
    while (this.running) {
      const started = performance.now();
      const source = this.source;
      if (!source || this.paused()) {
        await sleep(150);
        continue;
      }
      let bitmap: ImageBitmap | null = null;
      try {
        bitmap = await source.grab();
      } catch (e) {
        this.logger.warn('camera', 'frame grab failed', e instanceof Error ? e.message : e);
      }
      if (!bitmap) {
        await sleep(100);
        continue;
      }
      const guide = this.guideFor(bitmap.width, bitmap.height);
      const keepFrame = this.wantSnapshot;
      this.wantSnapshot = false;
      const result = await this.send(bitmap, guide, this.shouldReadName(), keepFrame);
      if (result) this.handle(result, keepFrame);
      const elapsed = performance.now() - started;
      const wait = this.settings().minFrameInterval - elapsed;
      if (wait > 0) await sleep(wait);
    }
    this.logger.info('app', 'scan loop stopped');
  }

  private shouldReadName(): boolean {
    if (this.nameInFlight) return false;
    if (this.settings().alwaysReadNames) return true;
    const best = this.fusion()?.best;
    // Names are only needed when the catalog has no name for the card.
    return !!best && !best.resolution?.catalogName && !this.namedKeys.has(best.identity.key);
  }

  private handle(r: FrameResult, keepFrame: boolean): void {
    const now = performance.now();
    if (this.lastFrameAt) {
      const inst = 1000 / Math.max(1, now - this.lastFrameAt);
      this.fps.update((f) => (f ? f * 0.85 + inst * 0.15 : inst));
    }
    this.lastFrameAt = now;
    this.result.set(r);
    this.collectCardImage(r);
    this.fuser.setCatalog(this.catalog.index());

    let fusion: FusionState | null = null;
    if (!r.accepted) {
      this.framesWithoutCard++;
      if (this.framesWithoutCard === REARM_FRAMES) {
        if (!this.armed) this.logger.info('capture', 'card left the guide; ready for the next card');
        this.armed = true;
        this.fuser.reset();
        this.pendingName = null;
      }
      if (this.framesWithoutCard >= REARM_FRAMES || this.status() === 'reading') {
        this.status.set('searching');
        this.statusDetail.set(r.quad ? 'Card edges unclear: hold the card flat, inside the frame' : 'Hold a card inside the frame');
      }
    } else {
      this.framesWithoutCard = 0;
      if (r.nameCrop) void this.readName(r.nameCrop);
      fusion = this.fuser.push({ info: r.info, name: this.pendingName }, now);
      this.pendingName = null;
      this.fusion.set(fusion);
      this.applyFusion(fusion);
    }

    const summary: FrameSummary = {
      id: r.id,
      t: Math.round(now),
      accepted: r.accepted,
      score: r.scores ? round(r.scores.total) : null,
      read: r.info?.text ?? null,
      confidence: r.info ? round(r.info.confidence) : null,
      variant: r.variant,
      vote: fusion?.lastVote ?? (r.accepted ? '' : 'no card'),
      status: fusion?.status ?? 'none',
      timings: mapValues(r.timings, (v) => Math.round(v)),
    };
    this.history.update((h) => [summary, ...h].slice(0, HISTORY));
    this.logger.debug(
      'read',
      `#${r.id} ${r.accepted ? `card score=${summary.score}` : 'no card'}${r.info ? ` read="${r.info.text}" conf=${summary.confidence}${r.variant && r.variant !== 'normal' ? ` (${r.variant})` : ''}` : ''} ${summary.vote} [${summary.timings.total}ms]`,
    );
    if (keepFrame) {
      this.snapshot.set({ at: new Date().toISOString(), result: r, fusion, name: this.lastName() });
      this.logger.info('app', `snapshot taken of frame #${r.id}`);
    }
  }

  private applyFusion(f: FusionState): void {
    const best = f.best;
    if (f.status !== 'confirmed' || !best) {
      if (this.status() !== 'holding' || best?.identity.key !== this.lastCapturedKey) {
        this.status.set('reading');
        this.statusDetail.set(best ? `Reading… ${best.identity.setCode} ${best.identity.number}` : 'Reading the card…');
      }
      return;
    }
    const key = best.identity.key;
    if (!this.armed && key === this.lastCapturedKey) {
      this.status.set('holding');
      this.statusDetail.set('Already captured. Remove the card to scan the next one.');
      return;
    }
    const set = best.resolution?.set ?? this.catalog.index()?.setByCode(best.identity.setCode) ?? null;
    const capture: Capture = {
      id: ++this.captureSeq,
      identity: best.identity,
      setName: set?.name ?? null,
      imageUrl: this.catalog.index()?.imageUrl(best.identity.setCode, best.identity.number) ?? null,
      known: best.known,
      at: Date.now(),
      reason: f.reason,
    };
    this.armed = false;
    this.lastCapturedKey = key;
    this.fuser.reset();
    this.captures.update((list) => [capture, ...list].slice(0, 50));
    this.queueContribution(capture, f);
    void this.collection.add(best.identity);
    navigator.vibrate?.(60);
    this.status.set('captured');
    this.statusDetail.set(`Captured ${best.identity.setCode} ${best.identity.number}${best.identity.name ? ` · ${best.identity.name}` : ''}`);
    this.logger.info('capture', `captured ${key} ${best.identity.name ?? '(no name)'}: ${f.reason}`, {
      identity: best.identity,
      tally: f.tally,
      notes: best.resolution?.notes,
    });
  }

  /** Starts collecting what this capture tells us, for the learned catalog and training. */
  private queueContribution(capture: Capture, f: FusionState): void {
    this.flushContribution();
    if (!this.contribution.active()) return;
    const read = this.result()?.info;
    const fromCatalog = !!f.best?.resolution?.catalogName;
    this.pendingContribution = {
      framesLeft: 10,
      observation: {
        identity: capture.identity,
        text: read?.text ?? `${capture.identity.setCode} ${capture.identity.number}`,
        confidence: read?.confidence ?? 0,
        nameRead: fromCatalog ? null : capture.identity.name,
        nameConfidence: fromCatalog ? 0 : (this.lastName()?.confidence ?? 0),
        model: this.model()?.createdAt ?? null,
        image: null,
      },
    };
  }

  /** Attaches a card image from a frame that still shows the captured card. */
  private collectCardImage(r: FrameResult): void {
    const pending = this.pendingContribution;
    if (!pending) return;
    const key = pending.observation.identity;
    const sameCard = r.info?.parsed?.setCode === key.setCode && r.info.parsed.number === key.number;
    if (r.cardImage && sameCard) {
      pending.observation.image = r.cardImage;
      this.flushContribution();
    } else if (--pending.framesLeft <= 0) {
      this.flushContribution();
    }
  }

  private flushContribution(): void {
    const pending = this.pendingContribution;
    this.pendingContribution = null;
    if (pending) void this.contribution.record(pending.observation);
  }

  private async readName(crop: RGBAImage): Promise<void> {
    this.nameInFlight = true;
    const key = this.fusion()?.best?.identity.key ?? null;
    try {
      this.tesseract ??= import('../vision/tesseract-engine').then((m) => {
        this.logger.info('name', 'loading Tesseract for name OCR');
        return new m.TesseractEngine();
      });
      const engine = await this.tesseract;
      const { readName } = await import('../vision/reader');
      const name = await readName(engine, crop);
      this.lastName.set(name);
      this.pendingName = name.value;
      if (key && name.value) this.namedKeys.add(key);
      this.logger.debug('name', `name read "${name.value}" (${Math.round(name.confidence)}) in ${Math.round(name.ms)}ms`, name.attempts);
    } catch (e) {
      this.logger.warn('name', 'name OCR failed', e instanceof Error ? e.message : e);
    } finally {
      this.nameInFlight = false;
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function round(v: number): number {
  return Math.round(v * 100) / 100;
}

function mapValues<T extends Record<string, number>>(o: T, fn: (v: number) => number): T {
  return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, fn(v)])) as T;
}
