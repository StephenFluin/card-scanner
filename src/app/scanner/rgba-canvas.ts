import { Directive, effect, ElementRef, inject, input } from '@angular/core';
import type { RGBAImage } from '../vision/image';
import { drawRGBA } from './image-utils';

/** Renders an RGBA pixel buffer into the host <canvas>. */
@Directive({ selector: 'canvas[appRgba]' })
export class RgbaCanvas {
  readonly appRgba = input<RGBAImage | null | undefined>(null);
  private readonly el = inject<ElementRef<HTMLCanvasElement>>(ElementRef);

  constructor() {
    effect(() => {
      const img = this.appRgba();
      const canvas = this.el.nativeElement;
      if (img) drawRGBA(canvas, img);
      else canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    });
  }
}
