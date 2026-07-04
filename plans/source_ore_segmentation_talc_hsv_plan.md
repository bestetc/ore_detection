# Source Ore Segmentation + Talc HSV Candidate Plan

## Summary

Build the next project step around saved multiclass segmentation masks from `set_1`, `set_2`, and `set_3`. Do not use LITHOS or ORENeXt. The source model predicts ore/mineral classes plus `background`, where `background` means non-metallic matrix and may include talc. Talc remains a separate dummy output for the final system, but without supervised talc training data it is produced only by an HSV candidate notebook for expert review.

## Key Changes

- Create a legend-audited RGB-to-class mapping from `masks_human`, using the full parenthetical mineral name as the canonical target class, not the short code.
- Merge labels globally by full normalized name: e.g. `cpp (chalcopyrite)` and other chalcopyrite legends map to one `chalcopyrite` channel.
- Include `set_3` full-name classes in the global target list; classes unique to `set_3` stay valid channels but are treated as rare classes with class-balanced loss/metrics.
- Include `background` as a real segmentation channel. Binary ore masks must be derived by summing all non-background mineral channels only, not by summing every channel.
- Keep existing binary notebook `notebooks/01_train_source_binary_segmentation.ipynb` as a sanity baseline. The new ore-type workflow skips binary conversion and trains from saved multiclass masks.

## Implementation Plan

- Add source modules under `src/ore_detection/*` for:
  - reading `masks_colored`;
  - loading/auditing legend mappings from a config generated from `masks_human`;
  - converting each color mask to a saved one-hot tensor `[C, H, W]`;
  - deriving `metallic_binary = one_hot[non_background_channels].sum().clamp(0, 1)`.
- Save converted segmentation masks to `data_work/source_ore_type_masks/{set_N}/{split}/{stem}.pt` and metadata to `data_work/source_ore_type_masks/class_map.json`.
- Add `notebooks/02_train_source_ore_segmentation.ipynb`:
  - loads pre-saved one-hot masks;
  - returns batches as image `[B,3,H,W]` and mask `[B,C,H,W]`;
  - trains a multiclass segmentation model with logits `[B,C,H,W]`;
  - uses `CrossEntropyLoss` from `argmax(one_hot)` plus multiclass Dice from one-hot masks;
  - reports per-class Dice/IoU and a derived non-background binary Dice/IoU.
- Add a dummy `talc` output contract for final UI/export/reporting, but exclude it from source model classes and loss until real talc masks exist.
- Add `notebooks/03_create_talc_hsv_candidates.ipynb`:
  - scans every baseline image;
  - converts RGB to HSV and analyzes Value-channel percentiles;
  - tests percentile thresholds such as p1, p2, p5, p10, p15, p20;
  - displays original image, V channel, and candidate talc overlays in a loop for expert checking;
  - saves review decisions and selected thresholds for later supervised talc-mask creation.

## Test Plan

- Verify every RGB color present in `masks_colored` is mapped to exactly one full-name target class, including `background`.
- Verify merged class IDs are stable across `set_1`, `set_2`, and `set_3`.
- Verify saved one-hot masks have shape `[C,H,W]`, match image dimensions, and sum to `1` per pixel.
- Verify derived binary ore masks exclude `background`.
- Verify the new dataset loader returns `[B,3,H,W]` images and `[B,C,H,W]` masks.
- Smoke-test `02_train_source_ore_segmentation.ipynb` with training disabled and one small batch.
- Smoke-test the HSV talc notebook on baseline images and confirm the expert-review loop can display every image.

## Assumptions

- `background` is the canonical non-metallic class and can contain talc-like dark matrix regions.
- Talc is not a supervised class until expert talc masks exist.
- Exact RGB-to-class mappings must come from `masks_human` legends, with spelling corrections recorded explicitly in config.
- `magnetite`, `ordinary magnetite`, and `copper-bearing magnetite` remain separate channels unless the legend/config aliases them by the same full name.
