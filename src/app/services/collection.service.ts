import { isPlatformBrowser } from '@angular/common';
import { computed, effect, inject, PLATFORM_ID, Service, signal } from '@angular/core';
import type { CardIdentity } from '../vision/catalog';
import { AuthService } from './auth.service';
import { FirebaseService } from './firebase.service';
import { Logger } from './logger.service';

/** A card the user owns, keyed by its printed identity ("SVI-001"). */
export interface OwnedCard {
  key: string;
  setCode: string;
  number: string;
  total: string | null;
  name: string | null;
  count: number;
  firstAdded: number;
  lastAdded: number;
}

type StoredCard = Omit<OwnedCard, 'key'>;

const LOCAL_KEY = 'card-scanner.collection.v1';

/**
 * The user's collection. Stored in Firebase Realtime Database at
 * users/{uid}/cards/{key} when signed in, otherwise in localStorage. Cards
 * scanned while signed out are merged into the account on sign-in.
 */
@Service()
export class CollectionService {
  private readonly auth = inject(AuthService);
  private readonly firebase = inject(FirebaseService);
  private readonly logger = inject(Logger);
  private readonly browser = isPlatformBrowser(inject(PLATFORM_ID));

  private readonly entries = signal<Record<string, OwnedCard>>({});
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly mode = computed(() => (this.auth.user() ? 'cloud' : 'local'));
  readonly cards = computed(() =>
    Object.values(this.entries()).sort(
      (a, b) => a.setCode.localeCompare(b.setCode) || a.number.localeCompare(b.number),
    ),
  );
  readonly totalCards = computed(() => this.cards().reduce((n, c) => n + c.count, 0));
  readonly uniqueCards = computed(() => this.cards().length);

  private unsubscribe: (() => void) | null = null;

  constructor() {
    if (!this.browser) return;
    effect(() => {
      if (!this.auth.ready()) return;
      const user = this.auth.user();
      this.unsubscribe?.();
      this.unsubscribe = null;
      if (user) void this.attachCloud(user.uid);
      else this.entries.set(this.readLocal());
    });
  }

  count(key: string): number {
    return this.entries()[key]?.count ?? 0;
  }

  async add(card: CardIdentity): Promise<void> {
    const now = Date.now();
    const uid = this.auth.user()?.uid;
    this.logger.info('db', `add ${card.key} (${uid ? 'cloud' : 'local'})`);
    if (!uid) {
      this.updateLocal((all) => {
        const cur = all[card.key];
        all[card.key] = cur
          ? {
              ...cur,
              count: cur.count + 1,
              lastAdded: now,
              name: cur.name ?? card.name,
              total: cur.total ?? card.total,
            }
          : {
              key: card.key,
              setCode: card.setCode,
              number: card.number,
              total: card.total,
              name: card.name,
              count: 1,
              firstAdded: now,
              lastAdded: now,
            };
      });
      return;
    }
    await this.transact(uid, card.key, (cur) =>
      cur
        ? {
            ...cur,
            count: cur.count + 1,
            lastAdded: now,
            name: cur.name ?? card.name,
            total: cur.total ?? card.total,
          }
        : {
            setCode: card.setCode,
            number: card.number,
            total: card.total,
            name: card.name,
            count: 1,
            firstAdded: now,
            lastAdded: now,
          },
    );
  }

  /** Removes one copy; removes the card entirely when the count reaches zero. */
  async decrement(key: string): Promise<void> {
    const uid = this.auth.user()?.uid;
    this.logger.info('db', `decrement ${key}`);
    if (!uid) {
      this.updateLocal((all) => {
        const cur = all[key];
        if (!cur) return;
        if (cur.count <= 1) delete all[key];
        else all[key] = { ...cur, count: cur.count - 1 };
      });
      return;
    }
    await this.transact(uid, key, (cur) =>
      !cur || cur.count <= 1 ? null : { ...cur, count: cur.count - 1 },
    );
  }

  async increment(key: string): Promise<void> {
    const cur = this.entries()[key];
    if (cur) await this.add(cur);
  }

  async remove(key: string): Promise<void> {
    const uid = this.auth.user()?.uid;
    this.logger.info('db', `remove ${key}`);
    if (!uid) {
      this.updateLocal((all) => delete all[key]);
      return;
    }
    await this.transact(uid, key, () => null);
  }

  private async attachCloud(uid: string): Promise<void> {
    this.loading.set(true);
    try {
      const { db } = await this.firebase.get();
      const { ref, onValue } = await import('firebase/database');
      await this.mergeLocalInto(uid);
      const node = ref(db, `users/${uid}/cards`);
      this.unsubscribe = onValue(
        node,
        (snap) => {
          const val = (snap.val() ?? {}) as Record<string, StoredCard>;
          const next: Record<string, OwnedCard> = {};
          for (const [key, c] of Object.entries(val))
            next[key] = { key, ...c, total: c.total ?? null, name: c.name ?? null };
          this.entries.set(next);
          this.loading.set(false);
          this.logger.debug('db', `collection synced: ${Object.keys(next).length} cards`);
        },
        (err) => {
          this.loading.set(false);
          this.error.set(err.message);
          this.logger.error('db', 'collection listener failed', err.message);
        },
      );
    } catch (e) {
      this.loading.set(false);
      this.error.set(e instanceof Error ? e.message : String(e));
      this.logger.error('db', 'could not attach to cloud collection', e);
    }
  }

  /** Moves cards scanned while signed out into the signed-in account. */
  private async mergeLocalInto(uid: string): Promise<void> {
    const local = this.readLocal();
    const keys = Object.keys(local);
    if (keys.length === 0) return;
    this.logger.info('db', `merging ${keys.length} locally scanned cards into account`);
    for (const key of keys) {
      const l = local[key];
      await this.transact(uid, key, (cur) =>
        cur
          ? {
              ...cur,
              count: cur.count + l.count,
              lastAdded: Math.max(cur.lastAdded, l.lastAdded),
              name: cur.name ?? l.name,
            }
          : {
              setCode: l.setCode,
              number: l.number,
              total: l.total,
              name: l.name,
              count: l.count,
              firstAdded: l.firstAdded,
              lastAdded: l.lastAdded,
            },
      );
    }
    this.writeLocal({});
  }

  private async transact(
    uid: string,
    key: string,
    fn: (cur: StoredCard | null) => StoredCard | null,
  ): Promise<void> {
    try {
      const { db } = await this.firebase.get();
      const { ref, runTransaction } = await import('firebase/database');
      await runTransaction(ref(db, `users/${uid}/cards/${key}`), (cur: StoredCard | null) => {
        const next = fn(cur);
        if (!next) return null;
        // RTDB rejects undefined and our rules reject nulls for optional fields: drop them.
        return Object.fromEntries(
          Object.entries(next).filter(([, v]) => v !== null && v !== undefined),
        );
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.error.set(msg);
      this.logger.error('db', `write ${key} failed`, msg);
    }
  }

  private readLocal(): Record<string, OwnedCard> {
    try {
      const raw = localStorage.getItem(LOCAL_KEY);
      return raw ? (JSON.parse(raw) as Record<string, OwnedCard>) : {};
    } catch {
      return {};
    }
  }

  private writeLocal(all: Record<string, OwnedCard>): void {
    try {
      localStorage.setItem(LOCAL_KEY, JSON.stringify(all));
    } catch (e) {
      this.logger.warn('db', 'localStorage write failed', e);
    }
  }

  private updateLocal(fn: (all: Record<string, OwnedCard>) => void): void {
    const all = { ...this.readLocal() };
    fn(all);
    this.writeLocal(all);
    this.entries.set(all);
  }
}
