import sharp from 'sharp';
import type { GrayImage, RGBAImage } from '../../src/app/vision/image';

export async function loadRGBA(input: string | Buffer): Promise<RGBAImage> {
  const { data, info } = await sharp(input).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) };
}

export async function saveRGBA(img: RGBAImage, path: string): Promise<void> {
  await sharp(Buffer.from(img.data.buffer, img.data.byteOffset, img.data.length), {
    raw: { width: img.width, height: img.height, channels: 4 },
  }).toFile(path);
}

export async function saveGray(img: GrayImage, path: string): Promise<void> {
  const bytes = Buffer.alloc(img.width * img.height);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Math.max(0, Math.min(255, Math.round(img.data[i])));
  await sharp(bytes, { raw: { width: img.width, height: img.height, channels: 1 } }).toFile(path);
}

/** Round-trips through JPEG at the given quality, like a camera pipeline would. */
export async function jpegRoundTrip(img: RGBAImage, quality: number): Promise<RGBAImage> {
  const jpeg = await sharp(Buffer.from(img.data.buffer, img.data.byteOffset, img.data.length), {
    raw: { width: img.width, height: img.height, channels: 4 },
  })
    .jpeg({ quality })
    .toBuffer();
  return loadRGBA(jpeg);
}

export async function gaussianBlur(img: RGBAImage, sigma: number): Promise<RGBAImage> {
  if (sigma < 0.3) return img;
  const { data, info } = await sharp(Buffer.from(img.data.buffer, img.data.byteOffset, img.data.length), {
    raw: { width: img.width, height: img.height, channels: 4 },
  })
    .blur(sigma)
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) };
}
