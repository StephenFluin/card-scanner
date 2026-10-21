import type { CameraInfo } from './camera.service';
import type { LogEntry } from '../services/logger.service';
import type { FrameSummary, Snapshot, ScanSettings } from './scan-controller';
import type { FusionState } from '../vision/fusion';
import type { ModelInfo } from './scan-protocol';
import { grayToRGBA, rgbaToDataUrl } from './image-utils';

export interface DebugBundleInput {
  status: string;
  statusDetail: string;
  source: string;
  settings: ScanSettings;
  camera: CameraInfo | null;
  model: ModelInfo | null;
  catalog: { sets: number; generatedAt: string } | null;
  fps: number;
  fusion: FusionState | null;
  history: FrameSummary[];
  logs: LogEntry[];
  snapshot: Snapshot | null;
}

/**
 * Builds a single JSON file with everything needed to diagnose a scanning
 * problem offline: environment, settings, recent frames, logs and (if taken)
 * a snapshot with the full camera frame and intermediate images as PNG data URLs.
 */
export function buildDebugBundle(input: DebugBundleInput): Blob {
  const snap = input.snapshot;
  let snapshot: Record<string, unknown> | null = null;
  if (snap) {
    const { frame, nameCrop, debug, ...rest } = snap.result;
    snapshot = {
      at: snap.at,
      result: {
        ...rest,
        debug: debug ? { roi: debug.roi, candidates: debug.candidates } : undefined,
      },
      fusion: snap.fusion,
      name: snap.name ? { ...snap.name, prepared: undefined } : null,
      images: {
        frame: frame ? rgbaToDataUrl(frame, 'image/jpeg') : null,
        cardPreview: debug?.preview ? rgbaToDataUrl(debug.preview) : null,
        infoLine: debug?.infoCrop ? rgbaToDataUrl(debug.infoCrop) : null,
        nameCrop: nameCrop ? rgbaToDataUrl(nameCrop) : null,
        namePrepared: snap.name ? rgbaToDataUrl(grayToRGBA(snap.name.prepared)) : null,
      },
    };
  }
  const bundle = {
    kind: 'card-scanner-debug-bundle',
    version: 1,
    createdAt: new Date().toISOString(),
    environment: {
      userAgent: navigator.userAgent,
      screen: { width: screen.width, height: screen.height, dpr: devicePixelRatio },
      hardwareConcurrency: navigator.hardwareConcurrency,
      url: location.href,
    },
    status: input.status,
    statusDetail: input.statusDetail,
    source: input.source,
    settings: input.settings,
    camera: input.camera,
    model: input.model,
    catalog: input.catalog,
    fps: input.fps,
    fusion: input.fusion,
    history: input.history,
    snapshot,
    logs: input.logs,
  };
  return new Blob([JSON.stringify(bundle, null, 1)], { type: 'application/json' });
}
