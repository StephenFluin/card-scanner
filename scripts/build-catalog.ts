/**
 * Builds public/catalog.json: the optional set/number/name reference data the
 * scanner uses to correct OCR reads and show friendly names.
 *
 *   npm run catalog
 *
 * Source: TCGdex (https://tcgdex.dev). Only silver-border era series are
 * included. Sets that aren't in TCGdex yet still scan fine; they just show
 * the name read from the card instead of a catalog name.
 */
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { Catalog, CatalogSet } from '../src/app/vision/catalog';

const API = 'https://api.tcgdex.net/v2/en';
const SERIES = ['sv', 'me'];
/** Sets in those series that don't use the standard card layout. */
const EXCLUDED_SETS = new Set(['mfb', '30th-c']);

interface TcgdexSetBrief {
  id: string;
  name: string;
}
interface TcgdexSet {
  id: string;
  name: string;
  releaseDate: string;
  serie: { id: string };
  abbreviation?: { official?: string };
  cardCount: { official: number; total: number };
  cards: { id: string; localId: string; name: string; image?: string }[];
}

async function get<T>(path: string, attempts = 5): Promise<T> {
  for (let i = 1; ; i++) {
    const res = await fetch(`${API}${path}`);
    if (res.ok) return (await res.json()) as T;
    if (i >= attempts || res.status < 500) throw new Error(`${path}: HTTP ${res.status}`);
    console.warn(`${path}: HTTP ${res.status}, retrying (${i}/${attempts})`);
    await new Promise((r) => setTimeout(r, 1000 * 2 ** i));
  }
}

async function main(): Promise<void> {
  const sets: CatalogSet[] = [];
  for (const series of SERIES) {
    const s = await get<{ sets: TcgdexSetBrief[] }>(`/series/${series}`);
    for (const brief of s.sets) {
      if (EXCLUDED_SETS.has(brief.id)) continue;
      const set = await get<TcgdexSet>(`/sets/${encodeURIComponent(brief.id)}`);
      const code = set.abbreviation?.official;
      if (!code) {
        console.warn(`skipping ${set.id} (${set.name}): no printed set code`);
        continue;
      }
      const cards: Record<string, string> = {};
      for (const c of set.cards) cards[c.localId] = c.name;
      const withImage = set.cards.find((c) => c.image);
      sets.push({
        id: set.id,
        code,
        name: set.name,
        series: set.serie.id,
        releaseDate: set.releaseDate,
        official: set.cardCount.official,
        total: set.cardCount.total,
        imageBase: withImage?.image ? withImage.image.replace(/\/[^/]+$/, '') : null,
        cards,
      });
      console.log(`${code.padEnd(4)} ${set.id.padEnd(8)} ${String(set.cards.length).padStart(4)} cards  ${set.name}`);
    }
  }
  sets.sort((a, b) => b.releaseDate.localeCompare(a.releaseDate));
  const catalog: Catalog = { generatedAt: new Date().toISOString(), source: 'tcgdex', sets };
  const out = join(import.meta.dirname, '..', 'public', 'catalog.json');
  await writeFile(out, JSON.stringify(catalog));
  console.log(`wrote ${out} (${sets.length} sets)`);
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
