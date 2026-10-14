/**
 * Builds public/catalog.json: the optional set/number/name reference data the
 * scanner uses to correct reads and show friendly names.
 *
 *   npm run catalog
 *
 * Set and card lists are fetched from a public card API; sets it doesn't list
 * yet (e.g. before release) can be added by hand in scripts/extra-sets.json.
 * Only silver-border era series are included. Sets missing from the catalog
 * still scan fine; they just show the name read from the card.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { Catalog, CatalogSet } from '../src/app/vision/catalog';

const API = 'https://api.tcgdex.net/v2/en';
const SERIES = ['sv', 'me'];
/** Sets in those series that don't use the standard card layout. */
const EXCLUDED_SETS = new Set(['mfb', '30th-c']);

interface ApiSetBrief {
  id: string;
  name: string;
}
interface ApiSet {
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
    const s = await get<{ sets: ApiSetBrief[] }>(`/series/${series}`);
    for (const brief of s.sets) {
      if (EXCLUDED_SETS.has(brief.id)) continue;
      const set = await get<ApiSet>(`/sets/${encodeURIComponent(brief.id)}`);
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
  const extra = JSON.parse(await readFile(join(import.meta.dirname, 'extra-sets.json'), 'utf8')) as {
    sets: { id: string; code: string; name: string; series: string; releaseDate: string; official: number }[];
  };
  for (const e of extra.sets) {
    const existing = sets.find((s) => s.code === e.code);
    if (existing) {
      if (existing.official !== e.official) {
        console.warn(`extra-sets.json: ${e.code} printed total ${e.official} differs from the API's ${existing.official}; using the API`);
      }
      console.log(`${e.code} is now in the API; the extra-sets.json entry can be removed`);
      continue;
    }
    sets.push({ id: e.id, code: e.code, name: e.name, series: e.series, releaseDate: e.releaseDate, official: e.official, total: e.official, imageBase: null, cards: {} });
    console.log(`${e.code.padEnd(4)} ${e.id.padEnd(8)}    - cards  ${e.name} (added by hand)`);
  }
  sets.sort((a, b) => b.releaseDate.localeCompare(a.releaseDate));
  const catalog: Catalog = { generatedAt: new Date().toISOString(), sets };
  const out = join(import.meta.dirname, '..', 'public', 'catalog.json');
  await writeFile(out, JSON.stringify(catalog));
  console.log(`wrote ${out} (${sets.length} sets)`);
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
