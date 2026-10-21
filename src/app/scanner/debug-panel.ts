import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, output, signal } from '@angular/core';
import { CatalogService } from '../services/catalog.service';
import { LogCategory, Logger, LogLevel } from '../services/logger.service';
import { CameraService } from './camera.service';
import { buildDebugBundle } from './debug-bundle';
import { downloadBlob, grayToRGBA } from './image-utils';
import { RgbaCanvas } from './rgba-canvas';
import { ScanController } from './scan-controller';

export type SourceRequest =
  { kind: 'camera' } | { kind: 'photo'; file: File } | { kind: 'video'; file: File };

const LEVEL_ORDER: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

/** Everything needed to see what the scanner is doing and why. */
@Component({
  selector: 'app-debug-panel',
  imports: [RgbaCanvas, DecimalPipe],
  templateUrl: './debug-panel.html',
  styles: `
    :host {
      display: grid;
      gap: 0.75rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 0.75rem;
    }
    dl {
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 0.1rem 0.75rem;
      margin: 0;
      font-size: 0.85rem;
    }
    dt {
      color: var(--text-muted);
    }
    dd {
      margin: 0;
      font-family: ui-monospace, Menlo, Consolas, monospace;
      word-break: break-word;
    }
    canvas {
      max-width: 100%;
      border: 1px solid var(--border);
      background: #000;
    }
    .info-crop {
      width: 352px;
      height: 64px;
    }
    .char {
      display: inline-block;
      min-width: 1ch;
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 1.1rem;
    }
    .scroll {
      max-height: 280px;
      overflow: auto;
    }
  `,
})
export class DebugPanel {
  protected readonly scan = inject(ScanController);
  protected readonly camera = inject(CameraService);
  protected readonly logger = inject(Logger);
  private readonly catalog = inject(CatalogService);

  readonly sourceRequest = output<SourceRequest>();

  protected readonly logLevel = signal<LogLevel>('info');
  protected readonly logCategory = signal<LogCategory | 'all'>('all');
  protected readonly recording = signal(false);
  protected readonly categories: LogCategory[] = [
    'app',
    'camera',
    'worker',
    'read',
    'name',
    'fusion',
    'capture',
    'catalog',
    'auth',
    'db',
  ];

  protected readonly logs = computed(() => {
    const min = LEVEL_ORDER[this.logLevel()];
    const cat = this.logCategory();
    return this.logger
      .entries()
      .filter((e) => LEVEL_ORDER[e.level] >= min && (cat === 'all' || e.category === cat))
      .slice(-300)
      .reverse();
  });

  protected readonly result = this.scan.result;
  protected readonly info = computed(() => this.result()?.info ?? null);
  protected readonly chars = computed(() => {
    const info = this.info();
    if (!info) return [];
    return info.text
      .split('')
      .map((c, i) => ({ c: c === ' ' ? '␣' : c, p: info.charConfidences[i] ?? 0 }));
  });
  protected readonly namePrepared = computed(() => {
    const n = this.scan.lastName();
    return n ? grayToRGBA(n.prepared) : null;
  });
  protected readonly catalogInfo = computed(() => {
    const idx = this.catalog.index();
    return idx ? { sets: idx.sets.length, generatedAt: idx.catalog.generatedAt } : null;
  });

  protected charColor(p: number): string {
    return p >= 0.9 ? 'var(--ok)' : p >= 0.5 ? 'var(--warn)' : 'var(--danger)';
  }

  protected setGuideFill(value: string): void {
    this.scan.settings.update((s) => ({ ...s, guideFill: Number(value) }));
  }

  protected setAlwaysReadNames(on: boolean): void {
    this.scan.settings.update((s) => ({ ...s, alwaysReadNames: on }));
  }

  protected setConsoleLevel(level: string): void {
    this.logger.consoleLevel.set(level as LogLevel);
  }

  protected onFile(event: Event, kind: 'photo' | 'video'): void {
    const el = event.target as HTMLInputElement;
    const file = el.files?.[0];
    el.value = '';
    if (file) this.sourceRequest.emit({ kind, file });
  }

  protected downloadBundle(): void {
    const blob = buildDebugBundle({
      status: this.scan.status(),
      statusDetail: this.scan.statusDetail(),
      source: this.scan.sourceLabel(),
      settings: this.scan.settings(),
      camera: this.camera.info(),
      model: this.scan.model(),
      catalog: this.catalogInfo(),
      fps: this.scan.fps(),
      fusion: this.scan.fusion(),
      history: this.scan.history(),
      logs: this.logger.snapshot(),
      snapshot: this.scan.snapshot(),
    });
    downloadBlob(blob, `card-scanner-debug-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    this.logger.info('app', 'debug bundle downloaded');
  }

  /** Records a short clip of the raw camera stream, for reproducing problems offline. */
  protected recordClip(seconds = 5): void {
    const stream = this.camera.stream();
    if (!stream || this.recording() || typeof MediaRecorder === 'undefined') return;
    const type = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'].find((t) =>
      MediaRecorder.isTypeSupported(t),
    );
    const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 12_000_000 });
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = () => {
      this.recording.set(false);
      const blob = new Blob(chunks, { type: rec.mimeType });
      downloadBlob(
        blob,
        `card-scanner-clip-${Date.now()}.${rec.mimeType.includes('mp4') ? 'mp4' : 'webm'}`,
      );
      this.logger.info(
        'camera',
        `recorded ${seconds}s clip (${Math.round(blob.size / 1024)} KiB, ${rec.mimeType})`,
      );
    };
    this.recording.set(true);
    rec.start();
    setTimeout(() => rec.stop(), seconds * 1000);
  }
}
