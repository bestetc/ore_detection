# Project Status

## Overall

Status: **trained binary and ore segmentation models are ready; Phase 5 is unblocked; UI work should follow `ui_interface.md`**.

Current active plan artifact:

- `.hermes/plans/2026-07-03_184532-color-mask-source-segmentation-ui-plan.md`

Latest verification:

- Command: `PYTHONPATH=src py -3.13 scripts/verify.py`
- Result: `87` tests passed under Python `3.13`.
- UI ad-hoc smoke: trained checkpoint prediction, HTTP index, HSV prediction editor, `/save-mask`, `/upload-image`, and browser load with no JS console errors passed.
- UI instrument rework smoke: editor exposes `ui_interface.md` instrument sections, top current-model/image metadata, grouped no-artifact view tools, active-learning add/remove brush controls, and HTTP `GET /` + `POST /predict` return the reworked UI.
- Notebook smoke-check: `notebooks/01_train_source_binary_segmentation.ipynb` executed top-to-bottom with `RUN_TRAINING = False`.

Default interpreter:

- Python `3.13` via `py -3.13`
- `.python-version`: `3.13`
- `pyproject.toml`: `requires-python = ">=3.13"`

---

## Completed / Partially Completed

### Phase 0 — Dataset Audit and Scale Metadata

Status: **mostly completed; source mask path must be `masks_colored`**.

Completed files:

- `configs/datasets.yaml`
- `src/ore_detection/data/inventory.py`
- `scripts/audit_all_datasets.py`
- `docs/source_datasets.md`
- `tests/test_inventory.py`

Audit result:

- baseline: `1222` images
- set_1: `84` image/mask pairs
- set_2: `49` image/mask pairs
- set_3: `47` image/mask pairs

Still needs check:

- Source dataset licenses/citations.
- Visual alignment review of source image + binary mask overlays.

### Phase 1 — Color Mask to Binary Mask Conversion

Status: **implemented for first binary source model**.

Completed files:

- `src/ore_detection/data/color_mask.py`
- `src/ore_detection/data/label_mapping.py`
- `src/ore_detection/data/mask_io.py`
- `scripts/convert_source_masks.py`
- `tests/test_color_mask.py`
- `tests/test_label_mapping.py`
- `tests/test_mask_io.py`

Dataset rule:

- Use `datasets/set_N/imgs` for OM images.
- Use `datasets/set_N/masks_colored` for training masks.
- Skip `datasets/set_N/masks`.
- Skip `datasets/set_N/masks_human`.

Generated binary masks:

- `data_work/binary_masks/set_1/`
- `data_work/binary_masks/set_2/`
- `data_work/binary_masks/set_3/`

Conversion result from `masks_colored`:

- set_1: `84` masks, ore fraction `0.691426`
- set_2: `49` masks, ore fraction `0.670051`
- set_3: `47` masks, ore fraction `0.506406`

Still retained for future R&D only:

- `data_work/mapped_masks/set_*/coarse/`
- `data_work/mapped_masks/set_*/species/`

Current first model target:

- `0 = background`
- `1 = ore`

### Phase 2 — Source Binary and Ore Segmentation Models

Status: **trained models are ready; checkpoints are stored under `models/`**.

Completed files:

- `configs/segmentation_source.yaml`
- `notebooks/01_train_source_binary_segmentation.ipynb`
- `notebooks/02_train_source_ore_segmentation.ipynb`
- `src/ore_detection/models/__init__.py`
- `src/ore_detection/models/simple_unet.py`
- `src/ore_detection/training/__init__.py`
- `src/ore_detection/training/source_dataset.py`
- `src/ore_detection/training/torch_dataset.py`
- `tests/test_source_dataset.py`
- `tests/test_torch_optional_pipeline.py`
- `tests/test_notebook_artifact.py`

Loader result:

- binary source samples: `180`
- train samples: `134`
- test samples: `46`
- set_1 train/test: `64/20`
- set_2 train/test: `37/12`
- set_3 train/test: `33/14`

Model artifacts:

- Binary ore/background segmentation: `models/source_binary_segmentation/001/best.pt`
- Ore/mineral multiclass segmentation: `models/source_ore_segmentation/001/best.pt`
- Each trained run may also include `last.pt` and `history.json`.

Notebook usage examples:

- Binary segmentation: `notebooks/01_train_source_binary_segmentation.ipynb`
- Ore segmentation: `notebooks/02_train_source_ore_segmentation.ipynb`
- Reuse code from the train notebooks for inference/model-loading patterns where practical.

PyTorch environment under Python `3.13`:

- `torch 2.7.1+cu128`
- notebook selected device: `cuda`

Current prediction policy:

- Use the binary model as the trusted ore outline predictor.
- Use the ore segmentation model for ore/mineral class identity and contact descriptors.
- The trained models unblock Phase 5 work.

### Visual QA for Source Masks

Status: **binary overlay tooling and HSV dummy prediction QA implemented; human review still needed**.

Completed files:

- `src/ore_detection/visualization/__init__.py`
- `src/ore_detection/visualization/overlay.py`
- `scripts/make_mask_qa_overlays.py`
- `scripts/generate_hsv_dummy_predictions.py`
- `tests/test_overlay.py`
- `tests/test_hsv_dummy_segmentation.py`

Generated artifacts:

- `data_work/qa_overlays/binary/`
- Initial binary QA overlays written: `9` total, `3` per source dataset.
- `data_work/predictions/hsv_dummy_smoke/`
- HSV dummy smoke predictions written: `3` source test images.

Still needs check:

- Human review of binary overlays for mask alignment and black-background assumption mistakes.

### Phase 3 — Backend/UI Prediction Store

Status: **updated local backend/UI now supports trained model selection, talc mask creation, brush editing, and active-learning one-hot mask saves**.

Completed files:

- `src/ore_detection/segmentation/__init__.py`
- `src/ore_detection/segmentation/hsv_dummy.py`
- `src/ore_detection/inference/__init__.py`
- `src/ore_detection/inference/prediction_store.py`
- `src/ore_detection/backend/__init__.py`
- `src/ore_detection/backend/ui_annotation.py`
- `src/ore_detection/backend/service.py`
- `src/ore_detection/backend/app.py`
- `scripts/run_backend_ui.py`
- `run_ui.cmd`
- `run_ui.sh`
- `tests/test_prediction_store.py`
- `tests/test_prediction_correction_store.py`
- `tests/test_backend_service.py`
- `tests/test_backend_artifacts.py`

Implemented:

- HSV Value-channel dummy binary mask.
- Bright foreground mode for ore-like regions.
- Dark foreground mode for talc/dark-region candidates.
- Optional standard scaling preprocessing before HSV thresholding.
- UI-readable artifacts:
  - `ore_mask.png`
  - `ore_confidence.png`
  - `overlay.png`
  - `metadata.json`
- Local backend UI form for model selection, image path selection, and drag/drop image upload.
- Prediction review page with three synchronized panels: raw image, raw image + mask, and mask only.
- Model choices:
  - trained binary + ore segmentation checkpoints under `models/`
  - HSV Value dummy fallback baseline
- Colored class-index mask display and legend from the UI/model class list.
- View-only controls: synchronized zoom, crop displayed area, and return to full view.
- Instrument panel now follows `ui_interface.md` sections:
  - `View — no new artifacts` with synchronized scale, crop visible area, and return-to-full-view controls.
  - `Active learning brush` with current class selector, brush size, add selected class, and remove-to-background controls.
  - `Save active-learning mask` with full one-hot tensor save.
- Live brush editing by selected class; selecting/erasing background overwrites previous class.
- UI-only active-learning classes: `talc`, `normal_ore`, `hard_ore`, plus model classes.
- Talc mask creation controls:
  - Histogram of raw image by HSV Value.
  - Histogram of raw image by `R + G + B`.
  - Slider threshold for selected metric.
  - Pixels below threshold become editable `talc` class.
- Live metrics for hard ore, normal ore, and talc mask area fractions.
- Active-learning save endpoint `/save-mask` writes:
  - `class_index_mask.png`
  - `mask_preview.png`
  - `one_hot_mask.pt`
  - `metadata.json`
- Index page can reload previously saved `data_work/active_learning_masks/*/class_index_mask.png` masks as the editable starting mask for a new prediction/editor session.
- Drag/drop upload endpoint `/upload-image` stores images under `data_work/ui_uploads/`.
- Safe artifact serving under `/artifact?path=...`.
- Safe project image serving under `/source-image?path=...`.
- Accept-current-mask action under `/accept`.
- Accepted correction persistence under `corrections/`:
  - `ore_mask.png`
  - `correction_metadata.json`

UI requirements source:

- User-maintained demand file: `ui_interface.md`.
- The user will add new UI demands to `ui_interface.md`.
- UI bug reports will be provided in this chat.

Current `ui_interface.md` demand status:

- Implemented model selection, image path selection, drag/drop upload, three-panel display, color legend, metrics, zoom/crop/full-view, brush editing, talc thresholding, histograms, and one-hot tensor save.
- Remaining UI hardening: browser-side polish after manual visual use, large-image performance tuning, and richer brush UX if needed.

Run UI:

```bash
./run_ui.sh
```

On Windows, double-click or run:

```cmd
run_ui.cmd
```

Smoke verification:

- `PYTHONPATH=src py -3.13 scripts/verify.py` passed `87` tests.
- Ad-hoc UI smoke passed:
  - trained checkpoint prediction on a real source image wrote binary and multiclass masks;
  - `GET /` returned model/image UI;
  - `POST /predict` returned editor with talc controls;
  - `POST /save-mask` wrote a torch one-hot tensor;
  - `POST /upload-image` stored a drag/drop image;
  - Browser-loaded editor page had no JS console errors.
- Ad-hoc reload-mask verification passed:
  - listed saved `class_index_mask.png` files;
  - index page showed saved-mask reload control;
  - prediction copied saved mask into artifacts as `loaded_class_index_mask.png`;
  - editor used loaded mask as initial editable mask;
  - HTTP `GET /` and `POST /predict` worked with `saved_mask_path`.
- Ad-hoc UI instrument rework verification passed:
  - rendered editor exposes `ui_interface.md` instrument sections;
  - top current-model and image-address metadata are visible;
  - view tools are grouped as no-artifact tools;
  - active-learning brush has explicit add/remove controls;
  - HTTP `GET /` and `POST /predict` return the reworked UI.

### Phase 4 — Descriptor Extraction: Morphology + Attachment/Contact

Status: **initial utilities implemented; incomplete vs full plan**.

Completed files:

- `src/ore_detection/descriptors/morphology.py`
- `src/ore_detection/descriptors/contacts.py`
- `tests/test_morphology_descriptors.py`
- `tests/test_contact_descriptors.py`

Implemented:

- connected components
- component area
- 4-neighbor perimeter
- `perimeter2_over_area`
- circularity
- bbox-fill solidity proxy
- small-component area fraction
- class-to-class contact length
- hetero-sulfide contact length

Still needs create/fix:

- descriptor aggregation
- true convex-hull solidity
- distance-transform width metrics
- skeleton length / area
- dark matrix contact fraction
- oxide contact fraction
- replacement-ring features
- confidence/artifact quality descriptors

### Phase 7 — Talc Strategy

Status: **talc threshold creation is implemented in the UI; reviewed talc remains manual/active-learning only**.

Completed files:

- `src/ore_detection/talc/candidates.py`
- `scripts/make_talc_candidates.py`
- `docs/talc_strategy.md`
- `tests/test_talc_candidates.py`

Current strategy:

- Talc appears as black/dark scattered regions in non-ore matrix.
- Auto-mask black/dark regions outside predicted ore mask.
- UI lets user select either HSV Value or `R+G+B` metric and set a threshold per image.
- Pixels below the selected threshold become editable `talc` class.
- Talc is now a UI annotation class in `docs/label_mapping.md`.
- Talc candidates must share the same one-class-per-pixel mask as ore classes.
- Use candidate mask for manual correction / active learning only.
- Do not train talc as blank-negative everywhere.

Still needs create/fix:

- validation set of corrected talc masks
- edit UX for previously reloaded active-learning masks after manual browser feedback
- improve manual browser UX after visual feedback

---

## Not Started

### Target 10x Sanity Validation

Status: **not started with dummy backend outputs**.

Next path:

- Run HSV dummy backend on baseline 10x crops.
- Review overlays in the UI.
- Tune HSV Value threshold and standard scaling before any ML work resumes.

### Phase 5 — Weakly Supervised Hard/Normal Classifier

Status: **unblocked by ready binary and ore segmentation models; implementation not started**.

Next needs:

- Run trained binary/ore model predictions on baseline crops.
- Generate descriptor table from stored predicted masks.
- Visually QA baseline predictions before trusting descriptors for hard/normal splitting.
- Build weak normal/hard classifier or clustering model from descriptor features.

### Phase 6 — Clustering and Active-Learning Selection

Status: **not started**.

Blocked by:

- Descriptor table not generated yet.
- Weak classifier not trained yet.

### Phase 8 — Panorama Inference and Aggregation

Status: **not started**.

Blocked by:

- Baseline prediction workflow using trained checkpoints.
- Descriptor aggregation on panorama/crop predictions.
- Weak classifier.

---

## Infrastructure Status

Completed:

- `pyproject.toml`
- `.python-version`
- package skeleton under `src/ore_detection/`
- dependency-light tests using stdlib `unittest`
- temporary canonical verification script: `scripts/verify.py`

Current verification command:

```bash
PYTHONPATH=src py -3.13 scripts/verify.py
```

Available under Python `3.13`:

- `torch 2.7.1+cu128`
- `numpy 2.4.4`
- `pandas 3.0.2`
- `sklearn 1.8.0`
- `cv2 4.13.0`
- `PIL 12.2.0`

Needs create/fix:

- decide whether to use `pytest` as canonical runner
- decide whether generated `data_work/` artifacts should be ignored or versioned
- no git repository detected in current folder

---

## Immediate Next Tasks

1. Run the UI manually and visually test brush/talc/reload editing on representative baseline crops.
2. Tune large-image browser performance if panoramas/crops are slow in canvas.
3. For Phase 5, run trained binary/ore predictions on baseline crops, generate descriptors, visually QA them, then build the weak hard/normal classifier or clustering model.
