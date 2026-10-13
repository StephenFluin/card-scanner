# Card Scanner: plan & status

Last updated 2026-10-13.

## How it works
- **Identity = what's printed on the card**: set code + collector number (+ printed total) from the bottom-left info line (`G [SVI EN] 001/198 ●`). Key format `SVI-001`. New sets work without anyone's data.
- **Detection** (`src/app/vision/detect.ts`): edge-line fitting inside an on-screen guide, scored on line support, aspect ratio, border-pair geometry and containment. It returns its top few distinct quads.
- **Reading** (`src/app/vision/pipeline.ts` `readCard`): our own info-line model (`InfoLineNet`, `training/`) reads the line. It tries hypotheses in order (best quad, alternatives, upside-down, border-expanded for low-contrast backgrounds) until the model is confident. The per-frame budget adapts to device speed.
- **Multi-frame fusion** (`fusion.ts`): 2 matching reads confirm a catalog card; 3 confirm a card not in the catalog.
- **Catalog** (`public/catalog.json`, TCGdex): optional; corrects near-misses and supplies names/images.
- **Tesseract** reads the name only for cards not in the catalog.
- **App**: scanner page (camera, overlay, last capture with Undo/+1, tips, debug panel), collection page (by set, progress, +/-). Firebase Auth (Google) + RTDB, or localStorage until configured or signed in.

## Results (synthetic camera frames, `npm run eval`)
- Detection on contrasting backgrounds: ~95% of frames within 1.5% corner error (median 0.2%). White/gray backgrounds mostly fail: the silver border is invisible. The border-expanded fallback recovers many of these; the UI recommends a dark surface.
- v3 model on **sets it never saw** (PBL, DRI), 8 frames per card: 93.8% captured correctly with catalog (0% wrong); 91.3% with no catalog at all (brand-new set), 1.3% wrong.
- Browser (`?image=` samples): ~11 fps, ~90 ms/frame on desktop; card captured within ~300 ms.

## Next
1. Finish the v4 comparison (real-glyph code swaps, model width 1 vs 1.5). Train the final model on all sets (`train-all-v4`), export it, and commit `public/models/`.
2. Fill in `src/app/firebase.config.ts`, enable Google sign-in and RTDB, then deploy `database.rules.json`.
3. Test on a phone: `npx ng serve --ssl --host 0.0.0.0` (add the LAN host to `angular.json` `allowedHosts`). Use the debug panel's "Download debug bundle" / "Record 5s clip" for anything that misbehaves; clips and photos can be replayed with `?video=` / `?image=`.
4. If real-world detection struggles: a learned edge/corner detector trained on synthetic frames, or a printed dark scanning mat.

## Commands
- `npm run catalog` · `npm run scans` · `npm run eval -- --n 150 [--sets PBL,DRI] [--fusion 8] [--no-catalog] [--detect-only]`
- Training pipeline: see `training/README.md`.
- Dev server: `npx ng serve`; reproduce with `http://localhost:4200/?image=<url>` or `?video=<url>` (put local samples in `public/debug-samples/`, which is gitignored).
