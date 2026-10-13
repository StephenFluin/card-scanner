/**
 * Helpers for the card name: picking it out of Tesseract's words and
 * comparing it with known names.
 */

export interface OcrWord {
  text: string;
  confidence: number;
  /** Glyph height in pixels, used to find the large name text. */
  height: number;
  /** Baseline / vertical centre, used to group words into lines. */
  y: number;
  x: number;
}

const NAME_STOPWORDS = new Set(['HP', 'BASIC', 'STAGE', 'STAGE1', 'STAGE2', 'TRAINER', 'ITEM', 'SUPPORTER', 'STADIUM', 'TOOL', 'POKEMON', 'POKÉMON', 'EVOLVES', 'FROM', 'ENERGY']);

/**
 * Picks the card name out of the words found in the name region: the name is
 * the largest text, so we take the line of tallest words and keep the
 * alphabetic ones.
 */
export function parseName(words: readonly OcrWord[]): string | null {
  const usable = words.filter((w) => /[A-Za-zÀ-ÿ]{2,}/.test(w.text) && w.confidence > 30);
  if (usable.length === 0) return null;
  const tallest = Math.max(...usable.map((w) => w.height));
  const anchor = usable.find((w) => w.height === tallest)!;
  const line = usable
    .filter((w) => Math.abs(w.y - anchor.y) < tallest * 0.6 && w.height > tallest * 0.55)
    .sort((a, b) => a.x - b.x)
    .map((w) => w.text.replace(/[^A-Za-zÀ-ÿ'’.\-]/g, ''))
    .filter((t) => t.length > 0 && !NAME_STOPWORDS.has(t.toUpperCase()));
  const name = line.join(' ').trim();
  return name.length >= 2 ? name : null;
}

export function normalizeName(s: string): string {
  return s
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');
}

export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  const prev = new Array<number>(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0];
    prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = prev[j];
      prev[j] = Math.min(prev[j] + 1, prev[j - 1] + 1, diag + (a[i - 1] === b[j - 1] ? 0 : 1));
      diag = tmp;
    }
  }
  return prev[b.length];
}

/**
 * 0..1 similarity between an OCR'd name and a known name. OCR often drops the
 * stylised "ex" suffix, so a read that matches the start of the name counts.
 */
export function nameSimilarity(read: string, known: string): number {
  const a = normalizeName(read);
  const b = normalizeName(known);
  if (!a || !b) return 0;
  if (a === b) return 1;
  const bCore = b.replace(/ex$/, '');
  const best = Math.min(levenshtein(a, b), levenshtein(a, bCore), levenshtein(a.replace(/ex$/, ''), bCore));
  return Math.max(0, 1 - best / Math.max(bCore.length, 3));
}
