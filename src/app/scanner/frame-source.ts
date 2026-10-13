/** Something the scanner can pull frames from: the live camera, an uploaded photo or a video file. */
export interface FrameSource {
  readonly kind: 'camera' | 'image' | 'video';
  readonly label: string;
  /** Current frame size, or null if not ready yet. */
  size(): { width: number; height: number } | null;
  /** A new ImageBitmap of the current frame (ownership passes to the caller), or null if not ready. */
  grab(): Promise<ImageBitmap | null>;
  dispose(): void;
}

/** Frames from a <video> element (live camera stream or a video file). */
export class VideoSource implements FrameSource {
  constructor(
    private readonly video: HTMLVideoElement,
    readonly kind: 'camera' | 'video',
    readonly label: string,
  ) {}

  size(): { width: number; height: number } | null {
    const { videoWidth: width, videoHeight: height } = this.video;
    return width && height ? { width, height } : null;
  }

  async grab(): Promise<ImageBitmap | null> {
    if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !this.size()) return null;
    return createImageBitmap(this.video);
  }

  dispose(): void {
    // The video element is owned by the component.
  }
}

/** A still photo, fed repeatedly so multi-frame confirmation still works. */
export class ImageSource implements FrameSource {
  readonly kind = 'image';

  private constructor(
    private readonly bitmap: ImageBitmap,
    readonly label: string,
  ) {}

  static async fromBlob(blob: Blob, label: string): Promise<ImageSource> {
    return new ImageSource(await createImageBitmap(blob), label);
  }

  size(): { width: number; height: number } {
    return { width: this.bitmap.width, height: this.bitmap.height };
  }

  grab(): Promise<ImageBitmap> {
    return createImageBitmap(this.bitmap);
  }

  dispose(): void {
    this.bitmap.close();
  }
}
