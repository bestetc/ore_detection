# Improve Source Ore Segmentation Training

## Summary

Upgrade source ore training to use 4x-downsampled images and masks, GPU-resident training data, GPU-side train-only augmentation, train-set standard scaling, `tqdm` progress, periodic test-loss evaluation, and checkpoints in `models/source_ore_segmentation`.

## Key Changes

- Add downsample preparation for both images and masks.
  - Create `scripts/prepare_source_ore_downsampled_dataset.py`.
  - Read source images from `datasets/set_N/imgs/<split>`.
  - Read saved one-hot masks from `data_work/source_ore_type_masks`.
  - Downsample images with bilinear interpolation.
  - Downsample masks with nearest-neighbor interpolation, then save compact `uint8` class-index masks.
  - Save outputs under `data_work/source_ore_downsampled`.
  - Use 4x downsampling adjusted to dimensions divisible by 4; current images become `848x636`.

- Save train-set normalization constants.
  - Calculate RGB mean/std from downsampled train images only, in `[0, 1]`.
  - Write constants to `src/ore_detection/training/const.py`.
  - Apply the same standard scaling to train and test tensors.
  - Do not run augmentation during evaluation.

- Add GPU training and augmentation utilities.
  - Cache downsampled train tensors on GPU when `USE_GPU_CACHE = True`.
  - Cache test tensors on GPU if memory allows; otherwise use pinned CPU batches with non-blocking transfer.
  - Train-only augmentation: horizontal flip, vertical flip, random `rot90`, scale factor `[0.5, 2.0]`, random crop to `IMAGE_SIZE`, brightness/contrast `[0.6, 1.4]`.
  - Use bilinear interpolation for images and nearest interpolation for masks.
  - Rebuild one-hot masks per batch on GPU for Dice loss.

- Update `notebooks/02_train_source_ore_segmentation.ipynb`.
  - Add `tqdm.auto`.
  - Keep `RUN_TRAINING = False` by default and remove any `RUN_TRAINING = True` override cell.
  - Add `EVAL_EVERY_N_EPOCHS = 1`, `USE_GPU_CACHE = True`, `USE_GPU_AUGMENTATION = True`.
  - Evaluate test loss every `EVAL_EVERY_N_EPOCHS`.
  - Save `best.pt` when test loss improves.
  - Save `last.pt` every epoch and `history.json`.

## Test Plan

- Unit-test downsample preparation:
  - Downsampled image and mask dimensions match.
  - Mask class IDs are preserved with nearest-neighbor resizing.
  - Train RGB stats are calculated from train split only.
- Unit-test GPU augmentation:
  - Output image/mask shapes remain stable.
  - Mask labels remain valid class indices.
  - Evaluation path applies normalization but no augmentation.
- Update notebook artifact tests for:
  - `tqdm.auto`.
  - `EVAL_EVERY_N_EPOCHS`.
  - `models/source_ore_segmentation`.
  - `best.pt` saved by lower test loss.
  - no `RUN_TRAINING = True`.
- Run `py -3.13 scripts\verify.py`.
- Smoke-run notebook 02 with `RUN_TRAINING = False`.

## Assumptions

- "Masks should be downsampling too" means all supervised masks are pre-downsampled alongside images before training, not resized lazily in the train loop.
- Best checkpoint selection is based on mean test loss.
- Training uses random 512x512 crops from downsampled images; evaluation uses full downsampled test images.
