# TrashNet visual gallery and object detection

Run the existing web app:

```sh
cd web
npm ci
npm run dev
```

- `http://127.0.0.1:5173/trashnet`: 120 original TrashNet photographs with the recorded baseline results from Qwen3-VL 2B, Gemma 4 E2B, and SmolVLM 256M. Filter by ground-truth material and outcome, select a photo to compare model scores, and inspect the confusion matrix. Accepted answers, raw top-1 and abstentions remain distinct. Summary cards always cover all 120 images, regardless of gallery filters.
- `http://127.0.0.1:5173/detect`: actual DETR bounding boxes over TrashNet photos. Twelve examples display recorded CPU inference immediately. Upload a photo or choose any of the 120 images and press **Run detection on this photo** for actual browser WASM inference. The first run downloads approximately 43 MB of model weights plus the runtime; photos remain in the browser. Stop terminates the worker, including pending downloads or inference. The display threshold filters existing boxes without rerunning the model.
- The gallery's **Detect objects in this photo** link opens the same image in Detect. Photos without a saved detection show an empty state until actual inference is run.

Both routes support Korean, English, Japanese, light/dark themes, keyboard navigation and mobile screens. Static hosting supports both recordings and browser detection; no native inference server is required for these two routes. The existing `/demo` page remains available for L2S1 image decisions through a local native model.

## What the examples show

The gallery uses the frozen [2026-09-25 benchmark](../benchmarks/trashnet-vision-20260925/REPORT.md), 20 images per material, with its original default acceptance thresholds. The photos and every model observation are matched by path, label and SHA-256. Clicking a photo does not run L2S1 again. The saved counts are Qwen3-VL 91/120 accepted correct, Gemma 64/120, and SmolVLM 0/120 with all 120 abstained. This balanced sample is not full-dataset or deployment accuracy.

Detect is a separate object detector, **not a new L2S1 box API**. It uses [Xenova/detr-resnet-50](https://huggingface.co/Xenova/detr-resnet-50/tree/8be7ab59ff663484ee9ba2e8d8f267330d5ad03e), revision `8be7ab59ff663484ee9ba2e8d8f267330d5ad03e`, Q8. Its COCO object categories such as bottle, bowl and backpack are distinct from TrashNet's material labels. TrashNet provides material labels, not ground-truth boxes for this demo. No localization accuracy is claimed. The recorded detector includes real false positives (for example, metal406.jpg produces backpack detections); no boxes or labels were manually corrected.

Coordinates are normalized `[xmin, ymin, xmax, ymax]` relative to the original image. The overlay scales with the image; the list shows original-image pixel coordinates. The display clips out-of-bounds boxes, rejects non-finite/reversed boxes, and shows detections scoring at least the chosen threshold. The stored minimum is 0.05; the default display threshold is 0.5. Scores are not calibrated correctness probabilities. Browser WASM outputs/timings may differ from the CPU recording.

## Reproduce the bundled data

Download the archive from the pinned [TrashNet source](https://github.com/garythung/trashnet/tree/6fa2b878c6c1b4304b91109070ce0edf9279bb31), then run:

```sh
python3 scripts/export_trashnet_demo.py /path/to/dataset-resized.zip
cd web
node scripts/record-detect.mjs
```

The exporter verifies the frozen archive SHA-256, verifies all 120 original image hashes, and exports the unchanged three-model baseline observations. It includes the MIT attribution in `web/static/trashnet/THIRD_PARTY_NOTICE.txt`. The detector script runs real CPU inference for the first two frozen images in each class, records model revision, model SHA-256, original dimensions, runtime, date and latency, and writes `detections.json`. It uses `/tmp/l2s1-detect-cache` by default; override `L2S1_DETECT_CACHE` if needed. Model weights are not checked in.

## Verification

```sh
cd web
npm run check
npm run lint
npm run build
npx playwright test tests/visual-examples.spec.ts
# Opt-in: downloads the pinned public model and performs real browser inference twice.
L2S1_TEST_DETECT_LIVE=1 npx playwright test tests/visual-examples.spec.ts
```

If necessary, set `PLAYWRIGHT_CHROMIUM_EXECUTABLE` to an installed Chromium executable. The regular suite verifies image/observation hashes, abstentions, filters, normalized box geometry, uploads, empty results, and mobile dark mode. The opt-in case requires a real bottle detection in the browser and repeats inference on a second photo.
