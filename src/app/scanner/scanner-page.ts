import { DatePipe } from '@angular/common';
import { afterNextRender, Component, computed, DestroyRef, effect, ElementRef, inject, signal, viewChild } from '@angular/core';
import { CollectionService } from '../services/collection.service';
import { Logger } from '../services/logger.service';
import { CameraService } from './camera.service';
import { DebugPanel, SourceRequest } from './debug-panel';
import { ImageSource, VideoSource } from './frame-source';
import { Capture, ScanController } from './scan-controller';
import { ScanOverlay } from './scan-overlay';

const DEBUG_KEY = 'card-scanner.debug';

@Component({
  selector: 'app-scanner-page',
  imports: [ScanOverlay, DebugPanel, DatePipe],
  providers: [ScanController],
  templateUrl: './scanner-page.html',
  styleUrl: './scanner-page.scss',
})
export class ScannerPage {
  protected readonly scan = inject(ScanController);
  protected readonly camera = inject(CameraService);
  protected readonly collection = inject(CollectionService);
  private readonly logger = inject(Logger);
  private readonly video = viewChild.required<ElementRef<HTMLVideoElement>>('video');

  protected readonly sourceKind = signal<'camera' | 'image' | 'video'>('camera');
  protected readonly photoUrl = signal<string | null>(null);
  protected readonly frameSize = signal<{ width: number; height: number } | null>(null);
  protected readonly guide = computed(() => {
    const s = this.frameSize();
    return s ? this.scan.guideFor(s.width, s.height) : null;
  });
  protected readonly aspect = computed(() => {
    const s = this.frameSize();
    return s ? `${s.width} / ${s.height}` : '3 / 4';
  });
  protected readonly viewportWidth = computed(() => {
    const s = this.frameSize();
    const ratio = s ? s.width / s.height : 0.75;
    return `min(100%, calc((100dvh - 230px) * ${ratio.toFixed(4)}))`;
  });
  protected readonly last = this.scan.lastCapture;
  protected readonly recent = computed(() => this.scan.captures().slice(1, 8));

  private videoUrl: string | null = null;
  private wakeLock: WakeLockSentinel | null = null;

  constructor() {
    this.scan.debug.set(readDebugFlag());
    effect(() => writeDebugFlag(this.scan.debug()));

    afterNextRender(() => {
      const v = this.video().nativeElement;
      const updateSize = () => {
        if (v.videoWidth && v.videoHeight) this.frameSize.set({ width: v.videoWidth, height: v.videoHeight });
      };
      v.addEventListener('loadedmetadata', updateSize);
      v.addEventListener('resize', updateSize);
      void this.start();
    });

    inject(DestroyRef).onDestroy(() => {
      this.scan.setSource(null);
      this.camera.stop();
      this.revokeUrls();
      void this.wakeLock?.release();
    });
  }

  /**
   * Starts the camera, or, for reproducing problems, a sample from the URL:
   * ?image=<url> or ?video=<url> (e.g. a photo or clip saved from the debug panel).
   */
  private async start(): Promise<void> {
    const params = new URLSearchParams(location.search);
    const image = params.get('image');
    const video = params.get('video');
    const url = image ?? video;
    if (url) {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        const file = new File([blob], url.split('/').pop() ?? 'sample', { type: blob.type });
        this.scan.debug.set(true);
        await this.onSourceRequest({ kind: image ? 'photo' : 'video', file });
        return;
      } catch (e) {
        this.logger.error('app', `could not load sample ${url}`, e instanceof Error ? e.message : e);
      }
    }
    await this.useCamera();
  }

  protected async useCamera(deviceId?: string): Promise<void> {
    this.revokeUrls();
    this.sourceKind.set('camera');
    const v = this.video().nativeElement;
    v.removeAttribute('src');
    v.loop = false;
    if (deviceId) await this.camera.switchTo(deviceId);
    else await this.camera.start();
    const stream = this.camera.stream();
    if (!stream) {
      this.scan.setSource(null);
      return;
    }
    v.srcObject = stream;
    try {
      await v.play();
    } catch (e) {
      this.logger.warn('camera', 'video.play() failed', e instanceof Error ? e.message : e);
    }
    this.scan.setSource(new VideoSource(v, 'camera', this.camera.info()?.label ?? 'camera'));
    void this.keepAwake();
  }

  protected async onSourceRequest(req: SourceRequest): Promise<void> {
    if (req.kind === 'camera') {
      await this.useCamera();
      return;
    }
    this.camera.stop();
    this.revokeUrls();
    const v = this.video().nativeElement;
    v.srcObject = null;
    if (req.kind === 'photo') {
      const source = await ImageSource.fromBlob(req.file, req.file.name);
      this.photoUrl.set(URL.createObjectURL(req.file));
      this.sourceKind.set('image');
      this.frameSize.set(source.size());
      this.scan.setSource(source);
    } else {
      this.videoUrl = URL.createObjectURL(req.file);
      this.sourceKind.set('video');
      v.src = this.videoUrl;
      v.loop = true;
      await v.play();
      this.scan.setSource(new VideoSource(v, 'video', req.file.name));
    }
  }

  protected onCameraChange(event: Event): void {
    void this.useCamera((event.target as HTMLSelectElement).value);
  }

  protected onZoom(event: Event): void {
    void this.camera.setZoom(Number((event.target as HTMLInputElement).value));
  }

  protected toggleDebug(): void {
    this.scan.debug.update((d) => !d);
  }

  protected undo(c: Capture): void {
    void this.scan.undo(c);
  }

  protected addAnother(c: Capture): void {
    void this.scan.addAnother(c);
  }

  /** Keeps the screen on while scanning (where supported). */
  private async keepAwake(): Promise<void> {
    try {
      if (!this.wakeLock && 'wakeLock' in navigator) this.wakeLock = await navigator.wakeLock.request('screen');
    } catch {
      // Not allowed (e.g. battery saver); scanning still works.
    }
  }

  private revokeUrls(): void {
    const photo = this.photoUrl();
    if (photo) URL.revokeObjectURL(photo);
    this.photoUrl.set(null);
    if (this.videoUrl) URL.revokeObjectURL(this.videoUrl);
    this.videoUrl = null;
  }
}

function readDebugFlag(): boolean {
  const param = new URLSearchParams(location.search).get('debug');
  if (param !== null) return param !== '0' && param !== 'false';
  try {
    return localStorage.getItem(DEBUG_KEY) === '1';
  } catch {
    return false;
  }
}

function writeDebugFlag(on: boolean): void {
  try {
    localStorage.setItem(DEBUG_KEY, on ? '1' : '0');
  } catch {
    // Storage unavailable; the flag just isn't remembered.
  }
}
