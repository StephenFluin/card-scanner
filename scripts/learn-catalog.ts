/**
 * Learns catalog data from what users scanned (observations/{set}/{number}/{uid}
 * in the Realtime Database, written by ContributionService) and writes the
 * consensus to scripts/learned-sets.json for review. `npm run catalog` then
 * merges it in, below the API and scripts/extra-sets.json.
 *
 *   npm run learn-catalog -- [--min-users 3] [--agreement 0.9] [--file observations.json]
 *
 * Reads with your Firebase CLI login (admin access), so no service account is
 * needed. Safeguards against learning the model's own mistakes:
 *  - a value needs several distinct users and a strong (confidence-weighted) majority;
 *  - cards reported as "Wrong card" by enough users are dropped;
 *  - a set code not already in the catalog stays `"approved": false` until you
 *    review it and flip the flag (and give it a name).
 */
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import type { Catalog } from '../src/app/vision/catalog';
import { normalizeName } from '../src/app/vision/parse';

const ROOT = join(import.meta.dirname, '..');
const OUT = join(import.meta.dirname, 'learned-sets.json');

const { values: args } = parseArgs({
  options: {
    'min-users': { type: 'string', default: '3' },
    agreement: { type: 'string', default: '0.9' },
    file: { type: 'string' },
  },
});

interface Obs {
  text: string;
  conf: number;
  at: number;
  total?: string;
  name?: string;
  nameConf?: number;
  rejected?: boolean;
  count?: number;
}
type Observations = Record<string, Record<string, Record<string, Obs>>>;

export interface LearnedCard {
  /** Consensus name read from the cards, or null if there isn't one yet. */
  name: string | null;
  users: number;
  rejected: number;
}

export interface LearnedSet {
  code: string;
  /** Set name; fill in by hand for new codes. */
  name: string | null;
  /** New codes must be reviewed and approved before they enter the catalog. */
  approved: boolean;
  /** Consensus printed total, or null. */
  official: number | null;
  observers: number;
  cards: Record<string, LearnedCard>;
}

export interface LearnedSets {
  generatedAt: string;
  minUsers: number;
  agreement: number;
  sets: LearnedSet[];
}

/** Value with the largest weight, if it has at least `share` of the total. */
function consensus<T>(weights: Map<T, number>, share: number): T | null {
  let total = 0;
  let best: T | null = null;
  let bestW = 0;
  for (const [k, w] of weights) {
    total += w;
    if (w > bestW) {
      best = k;
      bestW = w;
    }
  }
  return total > 0 && bestW / total >= share ? best : null;
}

async function loadObservations(): Promise<Observations> {
  if (args.file) return JSON.parse(await readFile(args.file, 'utf8')) as Observations;
  const project = (JSON.parse(await readFile(join(ROOT, '.firebaserc'), 'utf8')) as { projects: { default: string } }).projects.default;
  const json = execFileSync('firebase', ['database:get', '/observations', '--project', project], {
    encoding: 'utf8',
    maxBuffer: 512 * 1024 * 1024,
  });
  return (JSON.parse(json || 'null') ?? {}) as Observations;
}

async function main(): Promise<void> {
  const minUsers = parseInt(args['min-users']!, 10);
  const agreement = parseFloat(args.agreement!);
  const observations = await loadObservations();
  const catalog = JSON.parse(await readFile(join(ROOT, 'public', 'catalog.json'), 'utf8')) as Catalog;
  const previous: LearnedSets | null = existsSync(OUT) ? (JSON.parse(await readFile(OUT, 'utf8')) as LearnedSets) : null;

  const sets: LearnedSet[] = [];
  for (const [code, numbers] of Object.entries(observations).sort(([a], [b]) => a.localeCompare(b))) {
    const known = catalog.sets.find((s) => s.code === code);
    const prev = previous?.sets.find((s) => s.code === code);
    const observers = new Set<string>();
    const totals = new Map<number, number>();
    const cards: Record<string, LearnedCard> = {};
    for (const [number, byUser] of Object.entries(numbers)) {
      const names = new Map<string, number>();
      const spellings = new Map<string, Map<string, number>>();
      let users = 0;
      let rejected = 0;
      for (const [uid, o] of Object.entries(byUser)) {
        observers.add(uid);
        if (o.rejected) {
          rejected++;
          continue;
        }
        users++;
        if (o.total) totals.set(parseInt(o.total, 10), (totals.get(parseInt(o.total, 10)) ?? 0) + 1);
        if (o.name) {
          const norm = normalizeName(o.name);
          const w = Math.max(0.05, (o.nameConf ?? 50) / 100);
          names.set(norm, (names.get(norm) ?? 0) + w);
          const sp = spellings.get(norm) ?? new Map<string, number>();
          sp.set(o.name, (sp.get(o.name) ?? 0) + 1);
          spellings.set(norm, sp);
        }
      }
      if (users < minUsers || rejected / (users + rejected) >= 0.2) continue;
      const norm = consensus(names, agreement);
      const nameVoters = norm ? Object.values(byUser).filter((o) => !o.rejected && o.name && normalizeName(o.name) === norm).length : 0;
      const name = norm && nameVoters >= minUsers ? consensus(spellings.get(norm)!, 0) : null;
      cards[number] = { name, users, rejected };
    }
    const totalVotes = [...totals.values()].reduce((a, b) => a + b, 0);
    const official = totalVotes >= minUsers ? consensus(totals, agreement) : null;
    sets.push({
      code,
      name: prev?.name ?? known?.name ?? null,
      approved: prev?.approved ?? !!known,
      official,
      observers: observers.size,
      cards,
    });
  }

  const result: LearnedSets = { generatedAt: new Date().toISOString(), minUsers, agreement, sets };
  await writeFile(OUT, JSON.stringify(result, null, 1) + '\n');

  for (const s of sets) {
    const named = Object.values(s.cards).filter((c) => c.name).length;
    const flag = s.approved ? '' : '  <- NEW CODE: review, set a name and "approved": true';
    console.log(`${s.code.padEnd(4)} observers ${String(s.observers).padStart(3)}  total ${s.official ?? '?'}  cards ${Object.keys(s.cards).length} (${named} named)${flag}`);
  }
  console.log(`wrote ${OUT}; run \`npm run catalog\` to merge it into public/catalog.json`);
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
