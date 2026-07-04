# Intergrowth Hard/Normal Morphology Classification

## Summary

Add a post-segmentation intergrowth stage that converts predicted metallic ore masks into a separate hard/normal/talc/background mask. The classifier uses local mask morphology, not raw RGB, and preserves the existing ore/mineral prediction artifacts.

## Key Changes

- Add an intergrowth output contract:
  - `intergrowth_mask.png`: class-index mask using current UI-compatible IDs: `background=0`, `talc=3`, `normal_ore=4`, `hard_ore=5`, `ignore=255`.
  - `intergrowth_score.png`: hard-score map, `0=normal-like`, `255=hard-like`.
  - `intergrowth_confidence.png`: confidence from distance to the calibrated hard/normal threshold.
  - `intergrowth_metrics.json`: normal/hard/talc/background areas, hard fraction over metallic ore, image-level label.
- Apply strict label precedence:
  - `ignore` stays ignore.
  - reviewed/manual talc pixels stay talc.
  - metallic ore pixels become `normal_ore` or `hard_ore`.
  - all remaining pixels stay background.
  - Current talc mask can be empty/false; no automatic talc detection is added.
- Use local-window morphology:
  - default `window_size=128`, `stride=64`;
  - for small images, clamp window size to the image size;
  - classify each local window from the binary metallic ore mask and paste only its stable center region into the output mask.
- Compute hard/normal features per window:
  - component count density;
  - component area distribution;
  - dominant component fraction;
  - `perimeter / sqrt(area)`;
  - compactness/circularity;
  - bbox-fill solidity proxy;
  - boundary/contact density;
  - small-object fraction;
  - optional mineral-contact density when `ore_multiclass_mask.png` exists.
- Add a dependency-light calibrated classifier:
  - train/calibrate from baseline crop predictions;
  - supervised hard/normal targets come from `Normal ore` and `Hard ore` folders;
  - `Talc contained` crops are processed and reported, but not used as hard/normal targets unless reviewed intergrowth labels exist;
  - save model config to `models/intergrowth_classifier/001/classifier.json`.

## Workflow And UI

- Add a notebook `05_calibrate_intergrowth_hard_normal.ipynb`:
  - regenerate baseline predictions/descriptors for all baseline crops if missing;
  - compute local-window morphology features;
  - calibrate threshold/feature signs on baseline weak labels;
  - save classifier JSON and validation metrics.
- Add backend postprocess support:
  - `POST /jobs/{job_id}/intergrowth` runs hard/normal classification on an existing binary or ore prediction job.
  - If `ore_mask.png` exists, use it as metallic ore.
  - If only `ore_multiclass_mask.png` exists, derive metallic ore as `class_index != background_index`.
- Update Inference UI:
  - add `Run intergrowth classification` after prediction completes;
  - add layer selection for ore mask vs intergrowth mask/score/confidence;
  - show normal/hard/talc/background metrics for full image and visible crop.
- Update Active Learning UI:
  - show the intergrowth mask as a parallel review layer;
  - keep brush/talc tools from overwriting mineral prediction unless the user explicitly edits the active-learning mask.

## Test Plan

- Unit-test local-window grid coverage and stable-region pasting.
- Unit-test fragmented synthetic mask classifies as hard and compact blob classifies as normal.
- Unit-test label precedence: talc/background/ignore never become normal/hard.
- Unit-test binary-only and ore-multiclass-derived metallic masks.
- Unit-test saved artifacts and metadata for `intergrowth_mask`, score, confidence, and metrics.
- Unit-test UI route content for intergrowth button, layer selector, and metrics.
- Run `py -3.13 scripts\verify.py`.

## Assumptions

- Binary ore mask remains the preferred metallic-ore outline when available.
- Normal/hard classification is local-window based, not whole-image only.
- Talc is a separate manual/reviewed class; current automatic talc mask remains false.
- `Talc contained` baseline crops are useful diagnostics but are not reliable hard/normal supervision without reviewed normal/hard labels.
