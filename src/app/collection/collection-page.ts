import { Component, computed, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CatalogService } from '../services/catalog.service';
import { CollectionService, OwnedCard } from '../services/collection.service';

interface SetGroup {
  code: string;
  name: string;
  releaseDate: string;
  /** Printed set size, if the set is in the catalog. */
  official: number | null;
  /** Distinct cards owned within the printed set size (secret rares excluded). */
  ownedOfficial: number;
  unique: number;
  copies: number;
  cards: (OwnedCard & { displayName: string; imageUrl: string | null })[];
}

@Component({
  selector: 'app-collection-page',
  imports: [RouterLink],
  templateUrl: './collection-page.html',
  styleUrl: './collection-page.scss',
})
export class CollectionPage {
  protected readonly collection = inject(CollectionService);
  private readonly catalog = inject(CatalogService);
  protected readonly filter = signal('');

  protected readonly groups = computed<SetGroup[]>(() => {
    const idx = this.catalog.index();
    const q = this.filter().trim().toLowerCase();
    const bySet = new Map<string, SetGroup>();
    for (const c of this.collection.cards()) {
      const set = idx?.setByCode(c.setCode) ?? null;
      const displayName = set?.cards[c.number] ?? c.name ?? 'Unknown';
      if (
        q &&
        !`${c.setCode} ${c.number} ${displayName} ${set?.name ?? ''}`.toLowerCase().includes(q)
      )
        continue;
      let g = bySet.get(c.setCode);
      if (!g) {
        g = {
          code: c.setCode,
          name: set?.name ?? 'Set not in catalog yet',
          releaseDate: set?.releaseDate ?? '9999',
          official: set?.official ?? null,
          ownedOfficial: 0,
          unique: 0,
          copies: 0,
          cards: [],
        };
        bySet.set(c.setCode, g);
      }
      g.cards.push({ ...c, displayName, imageUrl: idx?.imageUrl(c.setCode, c.number) ?? null });
      g.unique++;
      g.copies += c.count;
      if (g.official !== null && parseInt(c.number, 10) <= g.official) g.ownedOfficial++;
    }
    return [...bySet.values()].sort(
      (a, b) => b.releaseDate.localeCompare(a.releaseDate) || a.code.localeCompare(b.code),
    );
  });

  protected onFilter(event: Event): void {
    this.filter.set((event.target as HTMLInputElement).value);
  }

  protected removeAll(card: OwnedCard & { displayName: string }): void {
    if (
      confirm(
        `Remove all ${card.count} cop${card.count === 1 ? 'y' : 'ies'} of ${card.setCode} ${card.number} ${card.displayName}?`,
      )
    ) {
      void this.collection.remove(card.key);
    }
  }
}
