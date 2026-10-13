# Card Scanner — implementation plan

Status: dependencies installed (`firebase`, `tesseract.js`, dev: `sharp`, `tsx`). Nothing else built yet.

## Data
- Card catalog source: TCGdex (`https://api.tcgdex.net/v2/en`). pokemontcg.io was returning 502.
- Scope: `sv` and `me` series (silver border), excluding `mfb`. Set abbreviation is in `abbreviation.official` (e.g. `SVI`, `PAL`, `MEG`).
- Images: `https://assets.tcgdex.net/en/{series}/{set}/{localId}/high.png` (600x825) and `low.webp` (245x337). Some promo/energy sets have no images, so they can only be matched by OCR.
- `scripts/build-catalog.ts` (run with tsx) fetches sets and cards, downloads `low.webp` into a gitignored cache, computes hashes with the same shared TS code the browser uses, and writes `public/catalog/catalog.json`.

## Storage
- Firebase Auth with Google as the only provider. Realtime Database at `users/{uid}/collection/{cardId}` storing `{count, firstScannedAt, lastScannedAt}`. No relational DB is needed.
- When signed out, captures go to localStorage and are merged into RTDB on sign-in.
- Config in `src/app/firebase.config.ts`, plus `database.rules.json` (only the owner can read or write).
- Server routes use `RenderMode.Client`, since camera and auth are browser-only.

## Recognition pipeline (pure TS on RGBA buffers, so it runs in a Web Worker and in Node tests)
1. Show a portrait guide rectangle in card aspect (63:88). Request the highest camera resolution available (ideally 1080p or more), because the bottom text is tiny.
2. Detection: downscale the guide region plus a margin, take the max per-channel gradient, then search a band along each side for edge peaks. Fit lines with RANSAC, keeping the top-K candidates per side. Score the 4-line combinations by aspect ratio, edge strength, and a uniform silver border strip just inside the edge, which picks the outer card edge over the inner frame. Intersect the lines to get a quad.
3. Rectify: DLT homography, then a supersampled bilinear warp to the canonical card.
4. Hash match: 64-bit pHash of the full card (inner 4% margin, which skips the rounded corners) and of the art window (about x 7.5–92.5%, y 9.7–47%). Also try a 180° rotation. Linear Hamming scan over the catalog. Take the top-K with a margin check.
5. OCR (tesseract.js, main thread, only when the quad is stable):
   - Bottom-left strip at about x 3–35%, y 91.5–95%. Regex `([A-Z]{3})\s*EN.*?(\d{1,3})\s*/\s*(\d{2,3})`; promos have no `/total`.
   - Name strip at about x 12–70%, y 2.5–13%. Fuzzy-match the name against the candidates.
6. Temporal confirmation: confirm a card when the same ID wins in at least 3 of the last 5 frames, or when OCR set+number exactly agrees with the hash top-K. After capturing, wait until the card leaves the frame (or a different card appears) before capturing it again. Offer undo and +1.

## Debug tooling
- Toggle with `?debug=1`, persisted to localStorage.
- Overlay: guide rectangle, candidate lines, and the quad colored by score.
- Panel: rectified image, OCR crops, raw text and confidence, top-5 matches with distances and thumbnails, per-stage timings, FPS, and camera settings/capabilities (torch, focus, zoom).
- Logger ring buffer with level and category filters, plus copy/download.
- Analyze an uploaded image or video instead of the camera. Download a debug bundle (frame, rectified image, JSON).
- Offline accuracy harness (`scripts/eval.ts`): warp `high.png` scans onto backgrounds with perspective, blur, noise, lighting changes, and JPEG artifacts, then report top-1 accuracy.

## ArUco
Deferred. Edge fitting inside the guide gives rotation and perspective without a mat. Fall back to a printable 4-marker mat (js-aruco2) if real-world edge detection proves flaky.

## Routes
`''` scanner, `collection` (grouped by set, counts, +/-, set completion).

## Phone testing
getUserMedia requires HTTPS: `ng serve --ssl --host 0.0.0.0`, and add the LAN host to `allowedHosts`.
