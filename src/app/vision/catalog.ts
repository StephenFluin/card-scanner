import { levenshtein, nameSimilarity } from './parse';

/**
 * Optional reference data about known sets. Recognition never *requires* the
 * catalog: the card's own printed set code + number is its identity. The
 * catalog is only used to correct OCR mistakes, confirm reads faster and show
 * friendly set names. Cards from sets that aren't in the catalog yet are
 * still recognised and saved.
 */
export interface CatalogSet {
  /** Data source id (e.g. TCGdex "sv01"); only used to build image URLs. */
  id: string;
  /** Printed set code, e.g. "SVI". */
  code: string;
  name: string;
  series: string;
  releaseDate: string;
  /** Printed set size (the number after the slash). */
  official: number;
  /** Total cards including secret rares. */
  total: number;
  /** Base URL for card images; append `/{number}/low.webp`. */
  imageBase: string | null;
  /** Collector number (as printed, e.g. "001") -> card name. */
  cards: Record<string, string>;
}

export interface Catalog {
  generatedAt: string;
  source: string;
  sets: CatalogSet[];
}

export interface CardIdentity {
  /** Stable id derived from the printed card itself, e.g. "SVI-001". */
  key: string;
  setCode: string;
  number: string;
  /** Printed set size, if known. */
  total: string | null;
  name: string | null;
}

export function cardKey(setCode: string, number: string): string {
  return `${setCode}-${number}`;
}

export interface Resolution {
  set: CatalogSet | null;
  /** Card name from the catalog, if the number exists in the set. */
  catalogName: string | null;
  setCode: string;
  number: string;
  total: string | null;
  /** Evidence strength 0..1 from catalog agreement. */
  agreement: number;
  notes: string[];
}

export class CatalogIndex {
  private readonly byCode = new Map<string, CatalogSet[]>();

  constructor(readonly catalog: Catalog) {
    for (const s of catalog.sets) {
      const list = this.byCode.get(s.code) ?? [];
      list.push(s);
      this.byCode.set(s.code, list);
    }
  }

  get sets(): readonly CatalogSet[] {
    return this.catalog.sets;
  }

  setByCode(code: string): CatalogSet | null {
    return this.byCode.get(code)?.[0] ?? null;
  }

  /**
   * Reconciles an OCR read with the catalog. A misread set code is corrected
   * when exactly one known code is within one edit and the printed set size
   * agrees.
   */
  resolve(
    setCode: string,
    number: string,
    total: string | null,
    name: string | null,
    alternates: readonly string[] = [],
  ): Resolution {
    const notes: string[] = [];
    const totalNum = total ? parseInt(total, 10) : null;
    let set = this.setByCode(setCode);
    let code = setCode;
    for (const alt of alternates) {
      if (set) break;
      set = this.setByCode(alt);
      if (set) code = alt;
    }
    if (!set) {
      const near = this.catalog.sets.filter(
        (s) => levenshtein(s.code, setCode) <= 1 && (totalNum === null || s.official === totalNum),
      );
      if (near.length === 1) {
        set = near[0];
        notes.push(`set code ${setCode} corrected to ${set.code}`);
        code = set.code;
      } else if (totalNum !== null) {
        const byTotal = this.catalog.sets.filter((s) => s.official === totalNum);
        if (byTotal.length === 1) {
          notes.push(`set ${byTotal[0].code} suggested by printed total ${total} (code read ${setCode})`);
        }
      }
    }
    let agreement = 0;
    let catalogName: string | null = null;
    if (set) {
      agreement += 0.3;
      if (totalNum !== null) {
        if (set.official === totalNum) agreement += 0.3;
        else notes.push(`printed total ${total} != catalog ${set.official} for ${set.code}`);
      }
      catalogName = set.cards[number] ?? null;
      if (catalogName) {
        agreement += 0.2;
        if (name) {
          const sim = nameSimilarity(name, catalogName);
          if (sim >= 0.6) agreement += 0.2;
          else notes.push(`name read "${name}" vs catalog "${catalogName}" (sim ${sim.toFixed(2)})`);
        }
      } else {
        notes.push(`number ${number} not in catalog for ${set.code}`);
      }
    } else {
      notes.push(`set code ${setCode} not in catalog`);
    }
    return { set, catalogName, setCode: code, number, total, agreement: Math.min(1, agreement), notes };
  }

  imageUrl(setCode: string, number: string): string | null {
    const set = this.setByCode(setCode);
    if (!set?.imageBase || !(number in set.cards)) return null;
    return `${set.imageBase}/${number}/low.webp`;
  }
}
