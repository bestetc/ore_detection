# Talc Strategy

Talc is not predicted in the active binary/ore segmentation pipeline.

## Current Policy

- `talc` remains a separate annotation class for the UI.
- No supervised talc model is trained until expert talc masks exist.
- Do not generate talc descriptors from RGB/HSV threshold candidates in the intergrowth descriptor pipeline.
- Do not train blank talc masks as negative examples unless an expert has confirmed the image is talc-free.

## Active Prediction Pipeline

The trained-model prediction path uses:

- binary segmentation for the trusted ore outline;
- multiclass ore segmentation for mineral labels clipped to the binary ore mask;
- intergrowth descriptors from ore geometry and mineral contacts only.

Talc is intentionally absent from these prediction artifacts and descriptor rows.

## UI Annotation Direction

The UI annotation layer should keep a single class per pixel:

- `background_matrix`
- `sulfide_ore`
- `oxide_magnetite_hematite`
- `talc`
- `ignore`

Future UI work should save corrected class-index masks and metadata. Reviewed talc pixels can later become supervised talc training data and talc-fraction descriptors.
