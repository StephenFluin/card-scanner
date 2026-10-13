/**
 * Regions on a Scarlet & Violet / Mega Evolution era card, in normalized card
 * coordinates (0..1 across the full physical card, borders included).
 * Calibrated against official scans of regular, full-art, trainer and promo
 * cards; the bottom-left info line is in the same place on all of them:
 *
 *   [G] [SVI EN] 001/198 ●
 *    |     |      |   |
 *    |     |      |   └─ printed set size (absent on promos)
 *    |     |      └───── collector number
 *    |     └──────────── set code, white on a black box ("EN" is smaller)
 *    └────────────────── regulation mark
 */
export interface CardRegion {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Whole info line: the input of the info-line model (keep in sync with training/common.py). */
export const INFO_LINE: CardRegion = { x: 0.02, y: 0.934, w: 0.37, h: 0.034 };

/**
 * Name area. Pokémon names sit to the right of the stage badge near the top;
 * trainer names sit a little lower on the left. One generous region covers both.
 */
export const NAME_REGION: CardRegion = { x: 0.03, y: 0.022, w: 0.66, h: 0.11 };

/** Resolution the name region is rectified at for Tesseract (pixels per card width). */
export const NAME_CARD_WIDTH = 840;

/** Info-line model input size (keep in sync with training/common.py). */
export const INFO_LINE_INPUT = { width: 248, height: 32 };
