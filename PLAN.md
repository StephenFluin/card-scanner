# Card Scanner: plan & status

Last updated 2026-10-13. Work in progress; the app is **not usable yet** (no scanner page or app shell is wired up).

## Approach (agreed)
- **Identity = what's printed on the card**: set code + collector number (+ printed total), from the bottom-left info line (`G [SVI EN] 001/198 ●`). Key format `SVI-001`. Works for new sets without any third-party data.
- **Our own model reads the info line** (`InfoLineNet`, `training/`): small fully-convolutional CTC recognizer, trained in PyTorch on synthetic camera crops generated from TCGdex scans, exported to `public/models/info-line.{json,bin}`, run by a pure-TS runtime (`src/app/vision/line-model.ts`). No ML library or cloud call at runtime.
- **Tesseract** only reads the *name*, and only for cards not in the catalog.
- **Catalog** (`public/catalog.json`, `npm run catalog`, TCGdex): optional, used to correct reads and show names/images.
- **Card detection**: own edge-line fitting inside an on-screen guide (`src/app/vision/detect.ts`), no AR markers.
- Firebase Auth (Google only) + Realtime Database `users/{uid}/cards/{key}`; localStorage when signed out, merged on sign-in.

## Done
- `src/app/vision/`: image ops, homography/warp, detection, pipeline, model runtime, reader, fusion (multi-frame vote), catalog index, Tesseract wrapper.
- `scripts/`: `build-catalog.ts`, `download-scans.ts` (4,761 scans into `.cache/scans`, ~375 MB, gitignored), `eval.ts` (end-to-end eval on synthetic frames; `--detect-only`, `--clean`, `--names`, `--sets`), `debug-detect.ts` (draws detection internals), `lib/synth.ts` (3D-perspective frame synthesis).
- `training/`: `dataset.py` (on-the-fly augmentation), `model.py`, `train.py`, `export.py` (BN folding, fp16, numerical check).
- App pieces written but **not yet wired into routes/UI**: `scanner/scan.worker.ts` + `scan-protocol.ts`, `scan-controller.ts` (loop, fusion, capture/re-arm, name OCR, history, snapshots), `camera.service.ts`, `frame-source.ts`, `scan-overlay.ts`, `debug-panel.{ts,html}`, `debug-bundle.ts`, `rgba-canvas.ts`; `services/` logger, firebase, auth, catalog, collection. `database.rules.json`, `firebase.json`, `src/app/firebase.config.ts` (fill in config).

## Current measurements
- **Detection** (`npm run eval -- --detect-only --n 80`): on dark/wood/clutter backgrounds 38/40 frames have corner error <1.5% of card width (median 0.21%). White/gray backgrounds mostly fail: the light silver border is invisible against them, so the inner frame gets picked. The UI should tell users to scan on a dark surface.
- **Model: not working yet.** The first run (32×176 input, region h=0.05) got 0% even on *clean* crops of seen sets after 6.7k steps (e.g. `PAL 077/193` → `PAF 067/193`; `DRI` → `ASC`). The run was stopped. Suspected causes, in order:
  1. Text is too small: digits are about 10 px tall in a 32 px input because the region has large margins. The set-code box spans y≈0.941–0.961 (measured), so the region should be about y 0.934–0.968 (h≈0.034) at 32×248, or a taller input.
  2. Augmentation is too hard too early (many samples illegible). Add a curriculum (start at difficulty ~0.3) and/or cap blur by resolution.
  3. Check for a pipeline bug: overfit a tiny batch first. The `PAL`→`PAF` style errors suggest it learns a code prior instead of reading glyphs.
  4. Data generation is CPU-bound (~185 ms/step on 8 cores). Pre-render a dataset to disk for fast iteration.
  Keep `INFO_LINE` / input size in sync between `training/common.py` and `src/app/vision/card-layout.ts` (the runtime reads them from the model manifest).

## Next steps
1. Fix and retrain the model (above). Target: >95% exact on held-out sets (`python train.py --holdout PBL,DRI`), then train the final model with `--holdout none` and run `python export.py runs/final/best.pt`.
2. Move the worker's read-with-fallbacks into shared code (normal → flipped 180° → quad expanded by the border width ≈4%/2.9% for low-contrast backgrounds) so `eval.ts` measures exactly what the app does.
3. Wire up the UI: app shell (nav, Google sign-in button, local-mode notice), lazy routes `''` → scanner page (video + `ScanOverlay` + status + last capture with Undo/+1 + `DebugPanel` via `?debug=1`), `collection` → collection page (grouped by set, counts, +/-, set progress). Set server routes to `RenderMode.Client`.
4. `ng build`, test in the browser with "Analyze photo", then on a phone (`ng serve --ssl --host 0.0.0.0`, add the host to `allowedHosts`).
5. Optionally self-host the Tesseract worker/core/lang files instead of using the jsdelivr CDN.

## Commands
- `npm run catalog`: rebuild the catalog. `npm run scans`: download training scans.
- `npm run eval -- --n 100 [--detect-only] [--clean] [--sets PBL,DRI] [--save-failures]`
- `npx tsx scripts/debug-detect.ts --n 12 --only-poor` → `.cache/detect/*.png`
- `cd training && OMP_NUM_THREADS=1 python3 train.py --holdout PBL,DRI` (logs/checkpoints in `training/runs/`, gitignored)
