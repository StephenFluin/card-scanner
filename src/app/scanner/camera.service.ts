import { inject, Service, signal } from '@angular/core';
import { Logger } from '../services/logger.service';

export type CameraState = 'idle' | 'requesting' | 'live' | 'denied' | 'unavailable' | 'error';

/** Track capabilities we care about (not all are in the DOM typings). */
interface ExtendedCapabilities extends MediaTrackCapabilities {
  torch?: boolean;
  zoom?: { min: number; max: number; step: number };
  focusMode?: string[];
}

export interface CameraInfo {
  label: string;
  width: number;
  height: number;
  frameRate: number | null;
  facingMode: string | null;
  focusMode: string | null;
  torch: boolean;
  zoom: { min: number; max: number; step: number; value: number } | null;
}

const PREFERRED_DEVICE = 'card-scanner.camera';

/** Owns the getUserMedia stream: permissions, device choice, resolution, torch and zoom. */
@Service()
export class CameraService {
  private readonly logger = inject(Logger);
  readonly state = signal<CameraState>('idle');
  readonly error = signal<string | null>(null);
  readonly stream = signal<MediaStream | null>(null);
  readonly devices = signal<MediaDeviceInfo[]>([]);
  readonly info = signal<CameraInfo | null>(null);
  readonly torchOn = signal(false);

  private get track(): MediaStreamTrack | null {
    return this.stream()?.getVideoTracks()[0] ?? null;
  }

  async start(deviceId?: string): Promise<void> {
    this.stop();
    if (!navigator.mediaDevices?.getUserMedia) {
      this.state.set('unavailable');
      this.error.set(window.isSecureContext ? 'This browser has no camera API.' : 'Camera access needs HTTPS (or localhost).');
      this.logger.error('camera', this.error()!);
      return;
    }
    this.state.set('requesting');
    this.error.set(null);
    const preferred = deviceId ?? this.readPreferred();
    const video: MediaTrackConstraints = {
      // High resolution matters: the info line text is only ~1.5mm tall.
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      frameRate: { ideal: 30 },
      ...(preferred ? { deviceId: { exact: preferred } } : { facingMode: { ideal: 'environment' } }),
    };
    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      } catch (e) {
        if (preferred && (e as DOMException).name === 'OverconstrainedError') {
          this.logger.warn('camera', 'saved camera not available, falling back to default');
          this.writePreferred(null);
          stream = await navigator.mediaDevices.getUserMedia({ video: { ...video, deviceId: undefined, facingMode: { ideal: 'environment' } }, audio: false });
        } else {
          throw e;
        }
      }
      this.stream.set(stream);
      this.state.set('live');
      await this.applyFocus();
      this.refreshInfo();
      this.devices.set((await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput'));
      this.logger.info('camera', `started ${this.info()?.label} ${this.info()?.width}x${this.info()?.height}`, this.info());
    } catch (e) {
      const err = e as DOMException;
      const denied = err.name === 'NotAllowedError' || err.name === 'SecurityError';
      this.state.set(denied ? 'denied' : err.name === 'NotFoundError' ? 'unavailable' : 'error');
      this.error.set(
        denied
          ? 'Camera permission was denied. Allow camera access for this site in your browser settings, then try again.'
          : `${err.name}: ${err.message}`,
      );
      this.logger.error('camera', 'getUserMedia failed', { name: err.name, message: err.message });
    }
  }

  stop(): void {
    this.stream()?.getTracks().forEach((t) => t.stop());
    this.stream.set(null);
    this.torchOn.set(false);
    if (this.state() === 'live') this.state.set('idle');
  }

  async switchTo(deviceId: string): Promise<void> {
    this.writePreferred(deviceId);
    await this.start(deviceId);
  }

  async setTorch(on: boolean): Promise<void> {
    const track = this.track;
    if (!track) return;
    try {
      await track.applyConstraints({ advanced: [{ torch: on } as MediaTrackConstraintSet] });
      this.torchOn.set(on);
      this.logger.info('camera', `torch ${on ? 'on' : 'off'}`);
    } catch (e) {
      this.logger.warn('camera', 'torch not supported', e);
    }
  }

  async setZoom(value: number): Promise<void> {
    const track = this.track;
    if (!track) return;
    try {
      await track.applyConstraints({ advanced: [{ zoom: value } as MediaTrackConstraintSet] });
      this.refreshInfo();
    } catch (e) {
      this.logger.warn('camera', 'zoom failed', e);
    }
  }

  /** Re-triggers autofocus (useful when the camera has locked focus on the background). */
  async refocus(): Promise<void> {
    await this.applyFocus(true);
  }

  private async applyFocus(force = false): Promise<void> {
    const track = this.track;
    const caps = track?.getCapabilities?.() as ExtendedCapabilities | undefined;
    if (!track || !caps?.focusMode) return;
    try {
      if (force && caps.focusMode.includes('single-shot')) {
        await track.applyConstraints({ advanced: [{ focusMode: 'single-shot' } as MediaTrackConstraintSet] });
      }
      if (caps.focusMode.includes('continuous')) {
        await track.applyConstraints({ advanced: [{ focusMode: 'continuous' } as MediaTrackConstraintSet] });
      }
    } catch (e) {
      this.logger.debug('camera', 'focus constraint rejected', e);
    }
  }

  private refreshInfo(): void {
    const track = this.track;
    if (!track) return;
    const s = track.getSettings() as MediaTrackSettings & { focusMode?: string; zoom?: number };
    const caps = (track.getCapabilities?.() ?? {}) as ExtendedCapabilities;
    this.info.set({
      label: track.label,
      width: s.width ?? 0,
      height: s.height ?? 0,
      frameRate: s.frameRate ?? null,
      facingMode: s.facingMode ?? null,
      focusMode: s.focusMode ?? null,
      torch: !!caps.torch,
      zoom: caps.zoom ? { ...caps.zoom, value: s.zoom ?? caps.zoom.min } : null,
    });
  }

  private readPreferred(): string | null {
    try {
      return localStorage.getItem(PREFERRED_DEVICE);
    } catch {
      return null;
    }
  }

  private writePreferred(id: string | null): void {
    try {
      if (id) localStorage.setItem(PREFERRED_DEVICE, id);
      else localStorage.removeItem(PREFERRED_DEVICE);
    } catch {
      // Storage unavailable (private mode): the choice just isn't remembered.
    }
  }
}
