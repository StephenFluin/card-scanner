import { isPlatformBrowser } from '@angular/common';
import { effect, inject, PLATFORM_ID, Service, signal } from '@angular/core';
import type { CardIdentity } from '../vision/catalog';
import type { RGBAImage } from '../vision/image';
import { AuthService } from './auth.service';
import { FirebaseService } from './firebase.service';
import { Logger } from './logger.service';

const PREF_KEY = 'card-scanner.contribute';

/** What a capture tells us about a card, for the learned catalog. */
export interface Observation {
  identity: CardIdentity;
  /** Raw info-line text the model read, e.g. "DLR 018/103". */
  text: string;
  /** Model confidence (weakest character), 0..1. */
  confidence: number;
  /** Name read from the card by OCR (only when the catalog had none). */
  nameRead: string | null;
  nameConfidence: number;
  /** Model version (manifest createdAt). */
  model: string | null;
  /** Straightened card image, for future training. */
  image: RGBAImage | null;
}

/**
 * Shares what signed-in users scan to improve recognition over time:
 *  - an observation per user per card at observations/{set}/{number}/{uid}
 *    (read text, printed total, OCR'd name), aggregated by `npm run learn-catalog`;
 *  - one straightened card image per capture in Firebase Storage under
 *    scans/{uid}/, labelled with what was read, as real-world training data.
 * Users can turn this off; it's remembered per device.
 */
@Service()
export class ContributionService {
  private readonly auth = inject(AuthService);
  private readonly firebase = inject(FirebaseService);
  private readonly logger = inject(Logger);
  private readonly browser = isPlatformBrowser(inject(PLATFORM_ID));

  /** Whether this device shares scans (on by default). */
  readonly enabled = signal(this.readPref());

  constructor() {
    effect(() => this.writePref(this.enabled()));
  }

  /** True when a contribution would actually be sent. */
  active(): boolean {
    return this.enabled() && this.firebase.enabled && !!this.auth.user();
  }

  async record(o: Observation): Promise<void> {
    const uid = this.auth.user()?.uid;
    if (!uid || !this.active()) return;
    const { setCode, number, key } = o.identity;
    try {
      const { db, storage } = await this.firebase.get();
      let imagePath: string | null = null;
      if (o.image) {
        const blob = await toJpeg(o.image, 0.85);
        imagePath = `scans/${uid}/${key}/${Date.now()}.jpg`;
        const { ref, uploadBytes } = await import('firebase/storage');
        await uploadBytes(ref(storage, imagePath), blob, {
          contentType: 'image/jpeg',
          customMetadata: { label: o.text, key, model: o.model ?? '', confidence: o.confidence.toFixed(3) },
        });
      }
      const { ref, runTransaction } = await import('firebase/database');
      await runTransaction(ref(db, `observations/${setCode}/${number}/${uid}`), (cur: { count?: number } | null) => {
        const next: Record<string, unknown> = {
          text: o.text,
          conf: Math.round(o.confidence * 1000) / 1000,
          at: Date.now(),
          count: (cur?.count ?? 0) + 1,
        };
        if (o.identity.total) next['total'] = o.identity.total;
        if (o.nameRead) {
          next['name'] = o.nameRead.slice(0, 80);
          next['nameConf'] = Math.round(o.nameConfidence);
        }
        if (o.model) next['model'] = o.model.slice(0, 64);
        if (imagePath) next['image'] = imagePath;
        return next;
      });
      this.logger.info('db', `contributed observation ${key}${imagePath ? ' + image' : ''}`);
    } catch (e) {
      this.logger.warn('db', `contribution for ${key} failed`, e instanceof Error ? e.message : e);
    }
  }

  /** The user said this capture was the wrong card: strong evidence against the read. */
  async reject(identity: CardIdentity): Promise<void> {
    const uid = this.auth.user()?.uid;
    if (!uid || !this.active()) return;
    try {
      const { db } = await this.firebase.get();
      const { ref, update } = await import('firebase/database');
      await update(ref(db, `observations/${identity.setCode}/${identity.number}/${uid}`), { rejected: true });
      this.logger.info('db', `marked ${identity.key} as a misread`);
    } catch (e) {
      this.logger.warn('db', `could not mark ${identity.key} as a misread`, e instanceof Error ? e.message : e);
    }
  }

  private readPref(): boolean {
    if (!this.browser) return false;
    try {
      return localStorage.getItem(PREF_KEY) !== '0';
    } catch {
      return true;
    }
  }

  private writePref(on: boolean): void {
    if (!this.browser) return;
    try {
      localStorage.setItem(PREF_KEY, on ? '1' : '0');
    } catch {
      // Storage unavailable; the preference just isn't remembered.
    }
  }
}

async function toJpeg(img: RGBAImage, quality: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  ctx.putImageData(new ImageData(new Uint8ClampedArray(img.data), img.width, img.height), 0, 0);
  return canvas.convertToBlob({ type: 'image/jpeg', quality });
}
