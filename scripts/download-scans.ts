/**
 * Downloads official card scans for every catalog card into .cache/scans and
 * writes a labels manifest. These are only used as *training data* for the
 * info-line recognition model; the app never needs them at runtime.
 *
 *   npm run scans
 */
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { Catalog } from '../src/app/vision/catalog';

const ROOT = join(import.meta.dirname, '..');
const OUT = join(ROOT, '.cache', 'scans');
const CONCURRENCY = 6;

/** Sets whose cards print only the number, without "/total". */
const NO_TOTAL = new Set(['SVP', 'MEP']);

export interface ScanLabel {
  file: string;
  setId: string;
  code: string;
  number: string;
  total: string | null;
  name: string;
  /** What the info line says, in the model's label format: "SVI 001/198". */
  label: string;
}

async function download(url: string, file: string): Promise<boolean> {
  for (let attempt = 1; attempt <= 4; attempt++) {
    const res = await fetch(url);
    if (res.ok) {
      await writeFile(file, Buffer.from(await res.arrayBuffer()));
      return true;
    }
    if (res.status === 404) return false;
    await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
  }
  return false;
}

async function main(): Promise<void> {
  const catalog = JSON.parse(
    await readFile(join(ROOT, 'public', 'catalog.json'), 'utf8'),
  ) as Catalog;
  const jobs: (ScanLabel & { url: string })[] = [];
  for (const set of catalog.sets) {
    if (!set.imageBase) continue;
    await mkdir(join(OUT, set.code), { recursive: true });
    const total = NO_TOTAL.has(set.code) ? null : String(set.official).padStart(3, '0');
    for (const [number, name] of Object.entries(set.cards)) {
      if (!/^\d{3}$/.test(number)) continue;
      jobs.push({
        url: `${set.imageBase}/${number}/high.webp`,
        file: join(set.code, `${number}.webp`),
        setId: set.id,
        code: set.code,
        number,
        total,
        name,
        label: total ? `${set.code} ${number}/${total}` : `${set.code} ${number}`,
      });
    }
  }
  const labels: ScanLabel[] = [];
  let next = 0,
    fetched = 0,
    missing = 0;
  await Promise.all(
    Array.from({ length: CONCURRENCY }, async () => {
      while (next < jobs.length) {
        const job = jobs[next++];
        const path = join(OUT, job.file);
        const ok =
          existsSync(path) || (await download(job.url, path).then((r) => (r && fetched++, r)));
        if (!ok) {
          missing++;
          continue;
        }
        const { url: _url, ...label } = job;
        labels.push(label);
        if ((fetched + missing) % 200 === 0 && fetched + missing > 0) {
          console.log(
            `${labels.length}/${jobs.length} (${fetched} downloaded, ${missing} missing)`,
          );
        }
      }
    }),
  );
  labels.sort((a, b) => a.file.localeCompare(b.file));
  await writeFile(join(OUT, 'manifest.json'), JSON.stringify(labels, null, 1));
  console.log(`done: ${labels.length} scans (${fetched} downloaded now, ${missing} without image)`);
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
