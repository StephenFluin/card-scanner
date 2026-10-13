import { isPlatformBrowser } from '@angular/common';
import { inject, PLATFORM_ID, Service, signal } from '@angular/core';
import { Catalog, CatalogIndex } from '../vision/catalog';
import { Logger } from './logger.service';

/**
 * Optional set/card reference data (public/catalog.json, built by
 * `npm run catalog`). Scanning works without it; it only corrects reads and
 * supplies names and images.
 */
@Service()
export class CatalogService {
  private readonly logger = inject(Logger);
  readonly index = signal<CatalogIndex | null>(null);
  readonly error = signal<string | null>(null);

  constructor() {
    if (isPlatformBrowser(inject(PLATFORM_ID))) void this.load();
  }

  private async load(): Promise<void> {
    try {
      const res = await fetch('catalog.json');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const catalog = (await res.json()) as Catalog;
      this.index.set(new CatalogIndex(catalog));
      const cards = catalog.sets.reduce((n, s) => n + Object.keys(s.cards).length, 0);
      this.logger.info('catalog', `loaded ${catalog.sets.length} sets, ${cards} cards (built ${catalog.generatedAt})`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.error.set(msg);
      this.logger.warn('catalog', `catalog unavailable, scanning without it: ${msg}`);
    }
  }
}
