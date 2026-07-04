Project task.
Develop an end-to-end local system for automated ore classification from panoramic optical microscopy (OM) images of polished sections.

## Current project status

The project now has trained source segmentation models and a post-training prediction/descriptor workflow.

Implemented:
- binary ore/background segmentation training notebook: `notebooks/01_train_source_binary_segmentation.ipynb`;
- multiclass ore/mineral segmentation training notebook: `notebooks/02_train_source_ore_segmentation.ipynb`;
- talc candidate review notebook kept for experiments: `notebooks/03_create_talc_hsv_candidates.ipynb`;
- trained-model prediction and intergrowth descriptor notebook: `notebooks/04_predict_ore_masks_and_descriptors.ipynb`;
- checkpoint inference helpers in `src/ore_detection/inference/model_prediction.py`;
- intergrowth descriptor aggregation in `src/ore_detection/descriptors/intergrowth.py`;
- baseline `SimpleUNet` and experimental CS-UNet-style model in `src/ore_detection/models/`;
- source downsample/prep and GPU training helpers in `src/ore_detection/training/`.

Verification command:

```bash
py -3.13 scripts/verify.py
```

The current expected verification result is all tests passing.

## Data

The `datasets` folder contains:

Baseline dataset:
- path: `datasets/baseline`;
- magnification: 10x;
- `panoramas`: huge resolution images expected as final production input;
- `Part N/<label>`: panorama crops with weak folder labels:
  - `Normal ore`: standard/coarse intergrowth regions;
  - `Hard ore`: thin or ragged intergrowth regions;
  - `Talc ore` or talc-bearing folders: regions with talc, possibly mixed with normal or hard ore.

Supervised source datasets:
- `datasets/set_1`: 50x OM images in `imgs`, colored pixel masks in `masks_colored`, train/test split.
  Classes: sphalerite, pyrite, galena, bornite, tennantite-tetrahedrite group, chalcopyrite minerals, background.
- `datasets/set_2`: 50x OM images in `imgs`, colored pixel masks in `masks_colored`, train/test split.
  Classes: pyrrhotite, chalcopyrite, pentlandite, magnetite, background.
- `datasets/set_3`: 50x OM images in `imgs`, colored pixel masks in `masks_colored`, train/test split.
  Classes: pyrite, arsenopyrite, covelline/covellite, bornite, chalcopyrite, ordinary magnetite, copper-bearing magnetite, hematite, background.

Do not use LITHOS for this project because it has no sulfide ore types.
Do not use ORENeXt as a core dependency because it targets ore rocks/particles, not reflected-light optical microscopy of polished sections.

## Model storage

Model artifacts are stored under `models/<model_family>/<serial>/`.

Current trained runs:
- binary source segmentation: `models/source_binary_segmentation/001/best.pt`;
- multiclass ore segmentation: `models/source_ore_segmentation/001/best.pt`.

Each serial folder may contain:
- `best.pt`;
- `last.pt`;
- `history.json`.

Training notebooks use `MODEL_SERIAL = '001'`. For another run of the same model family, create a new serial folder such as `002` and update `MODEL_SERIAL` before training.

## Current prediction policy

Use the trained binary model as the trusted ore outline predictor.

Prediction outputs:
- `ore_mask`: binary ore outline from the binary model;
- `ore_probability`: sigmoid probability map from the binary model;
- `ore_confidence`: binary-model confidence map;
- optional `ore_multiclass_mask`: mineral labels from the ore model, clipped to `ore_mask`;
- optional `ore_multiclass_confidence`: confidence for the selected mineral class;
- metadata with checkpoint path, serial, epoch, test metrics, class names, thresholds, and normalization stats.

The multiclass ore model enriches descriptors with mineral identity and mineral-contact information, but it should not override the binary ore outline for geometry.

## Talc policy

Talc is a separate UI/user-reviewed annotation class.

Do not:
- train talc as an always-negative dummy class;
- generate talc descriptors from RGB/HSV threshold candidates in the active intergrowth pipeline;
- treat dark-threshold candidates as proven talc.

Talc masks should come from reviewed UI annotations. Once real talc masks exist, they can become a supervised talc class and a talc-fraction descriptor.

## Intergrowth descriptors

Current descriptors are generated from binary and optional multiclass ore predictions:
- ore area and ore area fraction;
- component count and component area distribution;
- perimeter, perimeter density, `perimeter2_over_area`, circularity, bbox-fill proxy;
- small-component area fraction;
- ore/background contact length and density;
- mineral class area/fraction from multiclass masks;
- mineral-to-mineral contact lengths.

Weak normal/hard classification should be deferred until predicted masks are visually acceptable on baseline crops.

## Ontology and hypotheses

Relevant ontology lives at:

```text
../research_agent/brain/ore_structure
```

Important saved hypotheses:
- `H-20260702-63 - Human-Refined Zero-Shot Mask Stability`: relevant to active-learning UI and reviewed corrections.
- `H-20260702-65 - Spatial-Prior Mask Recovery Signal`: relevant to spatial priors and low-contrast matrix/talc review, but not active talc prediction.
- `H-20260703-10 - Microscopy Pretraining Transfer Gain`: relevant to CS-UNet and future architecture/pretraining experiments.
- `H-20260703-19 - Site-Specific SSL Descriptor Gain`: future self-supervised pretraining on unlabeled site images.
- `H-20260703-21 - Inclusion Anomaly Process Penalty`: directly relevant to intergrowth descriptors and hard/normal ore classification.
- `H-20260703-28 - Few-Shot Rare Texture Recovery Signal`: future rare texture support-set experiments.
- `H-20260703-30 - Boundary-Preserving Denoising Descriptor Gain`: future scratch/noise preprocessing experiments.

Useful concept files:
- `../research_agent/brain/ore_structure/brain/mineral-microscopy-image-datasets.md`;
- `../research_agent/brain/ore_structure/brain/deep-learning-segmentation-of-mineral-images.md`;
- `../research_agent/brain/ore_structure/brain/active-learning-microstructure-segmentation.md`;
- `../research_agent/brain/ore_structure/brain/microstructure-domain-adaptation.md`;
- `../research_agent/brain/ore_structure/brain/microstructure-visual-metrics.md`;
- `../research_agent/brain/ore_structure/brain/microstructure-foundation-segmentation.md`;
- `../research_agent/brain/ore_structure/brain/microscopy-denoising.md`.

## Code rules

- Use Python 3.13 via `py -3.13`.
- Put reusable functions in `src/ore_detection/*`.
- Use notebooks for EDA, hypothesis checks, visual review, and train/inference workflows.
- Keep notebook defaults non-mutating unless the user explicitly asks to run training/prediction.
- Prefer trained binary/ore model prediction for intergrowth work.
- Keep talc manual/UI-only until reviewed masks exist.
