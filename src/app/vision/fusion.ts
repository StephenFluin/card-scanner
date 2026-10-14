import { CardIdentity, CatalogIndex, cardKey, Resolution } from './catalog';
import type { InfoRead } from './reader';

/**
 * Combines info-line reads across frames. One frame can be misread, so a card
 * is only confirmed once the evidence agrees:
 *  - known card (card in the catalog, or its set in the catalog with a
 *    matching printed total): `votesKnown` matching frames,
 *    or a single frame whose set code, printed total, number and name all
 *    agree with the catalog;
 *  - unknown set or number (e.g. a brand new set): `votesUnknown` matching
 *    frames. New sets work without any catalog data.
 */

export interface FusionConfig {
  /** Reads older than this are forgotten. */
  maxAgeMs: number;
  /** Max reads kept. */
  window: number;
  votesKnown: number;
  votesUnknown: number;
  /** Reads whose weakest character is below this probability are ignored. */
  minConfidence: number;
}

export const DEFAULT_FUSION: FusionConfig = {
  maxAgeMs: 4000,
  window: 8,
  votesKnown: 2,
  votesUnknown: 3,
  minConfidence: 0.5,
};

export interface FusionInput {
  info: InfoRead | null;
  /** Card name read this frame, if name OCR ran. */
  name?: string | null;
}

export interface FusionCandidate {
  identity: CardIdentity;
  votes: number;
  resolution: Resolution | null;
  /** True when the catalog vouches for the card (card listed, or set listed with matching printed total). */
  known: boolean;
}

export interface FusionState {
  status: 'idle' | 'reading' | 'confirmed';
  best: FusionCandidate | null;
  /** All candidate keys with their votes, for the debug panel. */
  tally: { key: string; votes: number; known: boolean }[];
  reason: string;
  /** Why the latest frame did or didn't produce a vote. */
  lastVote: string;
}

interface Vote {
  t: number;
  key: string;
  setCode: string;
  number: string;
  total: string | null;
  name: string | null;
  resolution: Resolution | null;
}

export class ReadFusion {
  private votes: Vote[] = [];
  private names = new Map<string, string[]>();

  constructor(
    private catalog: CatalogIndex | null,
    private readonly cfg: FusionConfig = DEFAULT_FUSION,
  ) {}

  setCatalog(catalog: CatalogIndex | null): void {
    this.catalog = catalog;
  }

  reset(): void {
    this.votes = [];
    this.names.clear();
  }

  private toVote(input: FusionInput, now: number): Vote | string {
    const info = input.info;
    if (!info) return 'no info-line read';
    if (!info.parsed) return `unparseable "${info.text}"`;
    if (info.confidence < this.cfg.minConfidence) {
      return `low confidence ${info.confidence.toFixed(2)} for "${info.text}"`;
    }
    let { setCode } = info.parsed;
    const { number, total } = info.parsed;
    let resolution: Resolution | null = null;
    if (this.catalog) {
      resolution = this.catalog.resolve(setCode, number, total, input.name ?? null);
      setCode = resolution.setCode;
    }
    return { t: now, key: cardKey(setCode, number), setCode, number, total, name: input.name ?? null, resolution };
  }

  push(input: FusionInput, now: number): FusionState {
    this.votes = this.votes.filter((v) => now - v.t <= this.cfg.maxAgeMs);
    const vote = this.toVote(input, now);
    let lastVote: string;
    if (typeof vote === 'string') {
      lastVote = vote;
    } else {
      lastVote = `vote ${vote.key}${vote.resolution?.notes.length ? ` (${vote.resolution.notes.join('; ')})` : ''}`;
      this.votes.push(vote);
      if (this.votes.length > this.cfg.window) this.votes.shift();
      if (vote.name) {
        const list = this.names.get(vote.key) ?? [];
        list.push(vote.name);
        this.names.set(vote.key, list);
      }
    }
    return { ...this.evaluate(typeof vote === 'string' ? null : vote), lastVote };
  }

  private evaluate(latest: Vote | null): Omit<FusionState, 'lastVote'> {
    const byKey = new Map<string, Vote[]>();
    for (const v of this.votes) {
      const list = byKey.get(v.key) ?? [];
      list.push(v);
      byKey.set(v.key, list);
    }
    const tally = [...byKey.entries()]
      .map(([key, vs]) => ({ key, votes: vs.length, known: vouched(vs[vs.length - 1]) }))
      .sort((a, b) => b.votes - a.votes);
    if (tally.length === 0) {
      return { status: 'idle', best: null, tally, reason: 'no confident read yet' };
    }
    const top = tally[0];
    const vs = byKey.get(top.key)!;
    const last = vs[vs.length - 1];
    const catalogName = last.resolution?.catalogName ?? null;
    const identity: CardIdentity = {
      key: top.key,
      setCode: last.setCode,
      number: last.number,
      total: last.total,
      name: catalogName ?? mostCommon(this.names.get(top.key) ?? []),
    };
    const best: FusionCandidate = { identity, votes: top.votes, resolution: last.resolution, known: vouched(last) };

    const runnerUp = tally[1]?.votes ?? 0;
    if (runnerUp >= top.votes) {
      return { status: 'reading', best, tally, reason: `tie between ${top.key} and ${tally[1].key}` };
    }
    if (best.known) {
      if (top.votes >= this.cfg.votesKnown) {
        return { status: 'confirmed', best, tally, reason: `${top.votes} matching reads of a catalog card` };
      }
      if (latest?.key === top.key && (latest.resolution?.agreement ?? 0) >= 0.99) {
        return { status: 'confirmed', best, tally, reason: 'set code, total, number and name all match the catalog' };
      }
      return { status: 'reading', best, tally, reason: `catalog card, ${top.votes}/${this.cfg.votesKnown} reads` };
    }
    if (top.votes >= this.cfg.votesUnknown) {
      return { status: 'confirmed', best, tally, reason: `${top.votes} matching reads of a card not in the catalog` };
    }
    return { status: 'reading', best, tally, reason: `not in catalog, ${top.votes}/${this.cfg.votesUnknown} reads` };
  }
}

/**
 * Whether the catalog vouches for a read: the card itself is listed, or its set
 * is listed (e.g. added by hand before release) and the printed total matches.
 */
function vouched(v: Vote): boolean {
  const res = v.resolution;
  if (res?.catalogName) return true;
  return !!res?.set && !!v.total && res.set.official === parseInt(v.total, 10);
}

function mostCommon(values: string[]): string | null {
  const counts = new Map<string, number>();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
  let best: string | null = null;
  let n = 0;
  for (const [v, c] of counts) {
    if (c > n) {
      best = v;
      n = c;
    }
  }
  return best;
}
