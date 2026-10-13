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
- **Shipped model** (`public/models/info-line.*`, v4 trained on all sets), 8 frames per card: **95.0% captured correctly, 0% wrong**, 2.2 frames on average (normal difficulty); hard difficulty 90.8% / 0% wrong. Single frame: 90.5% read, 100% precision when confident. Detection + read ≈ 46 ms/frame on desktop.
- **Generalization to new sets** (v4 trained without PBL/DRI): 97.5% correct / 0% wrong with those sets in the catalog; **93.3% correct / 0.8% wrong with no catalog at all** (a brand-new set).
- Detection on contrasting backgrounds: ~95% of frames within 1.5% corner error (median 0.2%). On white/gray backgrounds the silver border is invisible; the border-expanded read hypothesis recovers most of them. Remaining misses are mostly there, so the UI recommends a dark surface.

## Next
1. Fill in `src/app/firebase.config.ts`, enable Google sign-in and RTDB, then deploy `database.rules.json`.
2. Test on a phone: `npm run start:phone`, then open `https://<this machine's LAN IP>:4200` and accept the self-signed certificate. Use the debug panel's "Download debug bundle" / "Record 5s clip" for anything that misbehaves; clips and photos can be replayed with `?video=` / `?image=`.
3. When a new set is released: `npm run catalog` (names/images), and optionally `npm run scans` plus a retrain (training/README.md) once scans exist. Scanning works before either.
4. If real-world detection struggles: a learned edge/corner detector trained on synthetic frames, or a printed dark scanning mat.

## Commands
- `npm run catalog` · `npm run scans` · `npm run eval -- --n 150 [--sets PBL,DRI] [--fusion 8] [--no-catalog] [--detect-only]`
- Training pipeline: see `training/README.md`.
- Dev server: `npx ng serve`; reproduce with `http://localhost:4200/?image=<url>` or `?video=<url>` (put local samples in `public/debug-samples/`, which is gitignored).
