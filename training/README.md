# Info-line model training

The scanner identifies a card by reading its bottom-left info line, e.g.
`G [SVI EN] 001/198 ●` → `SVI 001/198`, with a small model we train here
(`InfoLineNet`: fully convolutional, CTC, ~230k parameters). The browser runs it
with a pure TypeScript runtime (`src/app/vision/line-model.ts`); no ML library
or cloud service is involved at runtime.

## Why the training data is partly synthetic

Real scans contain only ~25 distinct set codes and as many printed set totals,
so a model trained on them alone memorizes codes and totals as whole words and
misreads new sets. Three augmentations force it to read characters instead:

| augmentation | what it does | file |
| --- | --- | --- |
| real-glyph code swap | cuts letters out of real set-code boxes and rebuilds random codes in the true card font | `codebox.py` (`extract_code_glyphs`, `swap_code_letters`) |
| font code box | renders a look-alike `XYZ EN` box in one of several condensed fonts (covers letters never seen in real codes) | `codebox.py` (`render_code_box`) |
| synthetic number | inpaints the printed `NNN/TTT` and renders a random one in a bold italic | `codebox.py` (`replace_number`) |
| negatives | misplaced or upside-down crops labelled empty, so the model says "nothing here" instead of inventing text | `dataset.py` |

Every sample is then degraded like a camera photo (resolution, blur, motion
blur, noise, JPEG, exposure, white balance, gradients, glare) and rectified
with a slightly wrong quad matching the measured error of `detect.ts`.

## Steps

```bash
npm run catalog        # public/catalog.json (sets, numbers, names)
npm run scans          # .cache/scans: official scans used as training data (~375 MB)
cd training
# Optional extra fonts (OFL, Google Fonts) for font-rendered codes, into ../.cache/fonts:
#   BarlowCondensed-{Bold,SemiBold,BoldItalic}, PT_Sans-Narrow-Web-Bold,
#   FiraSansCondensed-{Bold,BoldItalic}, SairaCondensed-Bold, EncodeSansCondensed-Bold

# Generalization experiment: hold out two whole sets.
python3 render.py --name train-holdout --per-card 100 --exclude PBL,DRI --synth-code 0.6 --synth-number 0.4 --negative 0.08
python3 render.py --name val-holdout --per-card 10 --include PBL,DRI --seed 7
python3 render.py --name val-seen-clean --per-card 1 --include SVI,PAL,TEF,ASC --clean
python3 train.py --train train-holdout --val val-holdout,val-seen-clean --name holdout --steps 40000

# Final model on every set, then export for the app.
python3 render.py --name train-all --per-card 100 --synth-code 0.6 --synth-number 0.4 --negative 0.08
python3 train.py --train train-all --val val-seen-clean --name final --steps 40000
python3 export.py runs/final/last.pt --note "all sets"
```

Then check the whole pipeline (detection → rectification → model) on synthetic
camera frames with `npm run eval -- --n 200`.

`render.py` writes uint8 arrays to `.cache/dataset/<name>/` (~3 GB per 400k
samples); `train.py` keeps them on the GPU, so an RTX 5090 trains at ~20 ms/step.

## Keep in sync

`common.py` (`INFO_LINE`, `INPUT_H`, `INPUT_W`, `CHARSET`) ↔
`src/app/vision/card-layout.ts`. The app reads region and size from the exported
manifest, so a re-export is enough after changing them.

## Real-world training images (contributed scans)

When a signed-in user captures a card (and "Share scans to improve recognition"
is on), the app uploads one straightened image of the card to Firebase Storage:

```
gs://card-scanner-234235.firebasestorage.app/scans/{uid}/{SET-NNN}/{timestamp}.jpg
```

Each image is 720 px wide (the info line is then about model-input resolution)
and carries metadata: `label` (what the model read, e.g. `DLR 018/103`), `key`,
`model` (model version) and `confidence`. The matching observation at
`observations/{SET}/{NNN}/{uid}` records `rejected: true` if the user pressed
**Wrong card**; exclude those images, or relabel them, before training.

Only the owning user can read their images; pull them with project-owner
credentials, e.g.:

```bash
gcloud auth login
gcloud storage cp -r gs://card-scanner-234235.firebasestorage.app/scans ../.cache/real-scans
```

These are real photos of straightened cards, so they can be fed through the
same rendering as the official scans (`dataset.py` crops the info line from a
full-card image) to fine-tune on real cameras, lighting and sleeves.
