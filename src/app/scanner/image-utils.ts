import type { GrayImage, RGBAImage } from '../vision/image';

/** Draws an RGBA buffer into a canvas (resizing the canvas to fit). */
export function drawRGBA(canvas: HTMLCanvasElement, img: RGBAImage): void {
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.putImageData(new ImageData(new Uint8ClampedArray(img.data), img.width, img.height), 0, 0);
}

export function rgbaToDataUrl(img: RGBAImage, type = 'image/png'): string {
  const canvas = document.createElement('canvas');
  drawRGBA(canvas, img);
  return canvas.toDataURL(type, 0.92);
}

export function grayToRGBA(g: GrayImage): RGBAImage {
  const data = new Uint8ClampedArray(g.width * g.height * 4);
  for (let i = 0; i < g.data.length; i++) {
    const v = g.data[i];
    data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  return { width: g.width, height: g.height, data };
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
