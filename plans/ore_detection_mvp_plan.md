# Ore Detection MVP Plan

## Summary

Build an MVP around supervised polished-section segmentation plus downstream ore-type classification. Use `set_1`, `set_2`, and `set_3` as the segmentation backbone because they contain pixel masks for sulfide/oxide ore minerals. Do not use LITHOS or ORENeXt for this project.

Talc remains an explicit output class, but there is no talc mask data yet. Add a dummy `talc` class in schemas, model heads, UI, exports, and metrics, while excluding it from supervised loss until talc annotations exist.

## Key Changes

- Train segmentation from `set_1`, `set_2`, and `set_3` with two label views:
  - dataset-specific mineral classes for pretraining and evaluation;
  - shared `ore_mineral` versus `background` target for baseline panoramas.
- Define production classes as:
  - `background`
  - `ore_mineral`
  - `talc_dummy`
  - `ignore`
- Keep `talc_dummy` available in the model interface and UI, but set training weight to `0` and report it as `not trained` until masks are added.
- Classify normal versus hard ore from predicted ore masks using morphology:
  - component area distribution
  - `perimeter / sqrt(area)`
  - compactness
  - solidity
  - boundary density
  - small-object fraction
  - ore/background contact length
- Calibrate morphology thresholds on the baseline crop folders: `Normal ore`, `Hard ore`, and `Talc contained`.
- Treat talc detection as a separate future module:
  - initial placeholder output is an empty talc mask with zero confidence;
  - UI allows expert drawing of talc masks;
  - corrected talc masks become active-learning data.

## Implementation Plan

1. Data audit
   Build manifests for baseline, `set_1`, `set_2`, and `set_3`; verify image-mask pairing, class IDs, train/test split, duplicate/leakage risk, and `set_3` rotated variants.

2. Label schema
   Create a unified class map and dataset-specific mappings. Convert RGB grayscale-style masks like `(6,6,6)` into integer labels.

3. Segmentation model
   Train a U-Net/DeepLab-style baseline with a shared binary ore head and optional dataset-specific mineral heads. Use grouped validation by original sample, not by augmented image.

4. Panorama inference
   Tile panoramas with overlap, stitch ore masks and confidence maps, and compute ore morphology descriptors per tile and panorama.

5. Normal/hard classifier
   Train or tune a lightweight classifier from mask-derived morphology using baseline crop labels. Avoid a single raw perimeter/area threshold as the only rule.

6. Talc placeholder
   Add talc as an output contract now, but do not claim talc model performance. Return empty masks until expert talc annotations exist.

7. UI and active learning
   Provide mask overlay, confidence map, class correction tools, talc drawing support, and save corrected masks for retraining.

## Test Plan

- Verify all image-mask pairs and dimensions.
- Check `set_3` augmentations do not leak across train/test.
- Unit-test mask color-to-ID conversion.
- Validate binary ore segmentation IoU/Dice.
- Validate morphology features against synthetic shapes and baseline crops.
- Report normal/hard F1 separately from segmentation metrics.
- Report talc as `not trained` until talc masks exist.

## Assumptions

- `set_1`, `set_2`, and `set_3` are allowed for model training.
- LITHOS and ORENeXt are excluded.
- Talc has no current supervised data.
- Code will later live under `src/*`; notebooks are for EDA, CV, and experiments only.
