import { Component, effect, ElementRef, input, viewChild } from '@angular/core';
import { INFO_LINE } from '../vision/card-layout';
import { lineEndpoints, Side } from '../vision/detect';
import { applyHomography, cardToImage } from '../vision/homography';
import type { Rect } from '../vision/image';
import type { FrameResult } from './scan-protocol';
import type { ScanStatus } from './scan-controller';

const SIDE_COLORS: Record<Side, string> = { left: '#ff6b6b', right: '#4dabf7', top: '#ffd43b', bottom: '#b197fc' };

/** Canvas drawn over the camera preview, in frame pixel coordinates. */
@Component({
  selector: 'app-scan-overlay',
  template: `<canvas #canvas aria-hidden="true"></canvas>`,
  styles: `
    :host,
    canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
  `,
})
export class ScanOverlay {
  readonly result = input<FrameResult | null>(null);
  readonly size = input<{ width: number; height: number } | null>(null);
  readonly guide = input<Rect | null>(null);
  readonly status = input<ScanStatus>('waiting');
  readonly debug = input(false);
  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');

  constructor() {
    effect(() => this.draw());
  }

  private draw(): void {
    const canvas = this.canvas().nativeElement;
    const size = this.size();
    const guide = this.guide();
    const r = this.result();
    const status = this.status();
    const debug = this.debug();
    if (!size) return;
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width = size.width;
      canvas.height = size.height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const unit = Math.max(size.width, size.height) / 400;
    // Results from an older frame size (camera switched) are skipped.
    const fresh = r && r.width === size.width && r.height === size.height ? r : null;

    if (guide) {
      // Dim everything outside the guide.
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.beginPath();
      ctx.rect(0, 0, size.width, size.height);
      roundRect(ctx, guide, unit * 6);
      ctx.fill('evenodd');
      ctx.restore();

      ctx.lineWidth = unit * 1.2;
      ctx.setLineDash(status === 'searching' || status === 'waiting' ? [unit * 6, unit * 4] : []);
      ctx.strokeStyle = status === 'captured' || status === 'holding' ? '#4ade80' : 'rgba(255,255,255,0.85)';
      ctx.beginPath();
      roundRect(ctx, guide, unit * 6);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (debug && fresh?.debug) {
      const d = fresh.debug;
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.lineWidth = unit * 0.5;
      ctx.strokeRect(d.roi.x, d.roi.y, d.roi.w, d.roi.h);
      for (const side of ['left', 'right', 'top', 'bottom'] as Side[]) {
        ctx.fillStyle = SIDE_COLORS[side];
        const vertical = side === 'left' || side === 'right';
        for (const p of d.points[side]) {
          const x = vertical ? p.s : p.t;
          const y = vertical ? p.t : p.s;
          ctx.fillRect(x - unit * 0.6, y - unit * 0.6, unit * 1.2, unit * 1.2);
        }
        ctx.strokeStyle = SIDE_COLORS[side];
        ctx.lineWidth = unit * 0.4;
        for (const line of d.candidates[side]) {
          const [a, b] = lineEndpoints(line, size.width, size.height);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    if (fresh?.quad) {
      const q = fresh.quad;
      ctx.lineWidth = unit * 1.6;
      ctx.strokeStyle = !fresh.accepted ? '#f87171' : fresh.info?.parsed ? '#4ade80' : '#fbbf24';
      ctx.beginPath();
      q.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
      ctx.stroke();
      if (fresh.accepted) {
        // Outline the info line we read, and mark the card's top-left corner.
        const H = cardToImage(q);
        const { x, y, w, h } = INFO_LINE;
        const corners = [
          { x, y },
          { x: x + w, y },
          { x: x + w, y: y + h },
          { x, y: y + h },
        ].map((p) => applyHomography(H, p));
        ctx.lineWidth = unit * 0.8;
        ctx.strokeStyle = '#7cc4ff';
        ctx.beginPath();
        corners.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.closePath();
        ctx.stroke();
        if (debug) {
          ctx.fillStyle = '#7cc4ff';
          ctx.beginPath();
          ctx.arc(q[0].x, q[0].y, unit * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }
}

function roundRect(ctx: CanvasRenderingContext2D, r: Rect, radius: number): void {
  ctx.moveTo(r.x + radius, r.y);
  ctx.arcTo(r.x + r.w, r.y, r.x + r.w, r.y + r.h, radius);
  ctx.arcTo(r.x + r.w, r.y + r.h, r.x, r.y + r.h, radius);
  ctx.arcTo(r.x, r.y + r.h, r.x, r.y, radius);
  ctx.arcTo(r.x, r.y, r.x + r.w, r.y, radius);
  ctx.closePath();
}
