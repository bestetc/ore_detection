# UI Inference Guide

This guide covers the CPU-only local inference workflow.

## 1. Install

From the repo root, run the CPU setup script.

Windows:

```bat
setup_ui_cpu.cmd
```

macOS/Linux:

```bash
./setup_ui_cpu.sh
```

## 2. Start The UI

Windows:

```bat
run_ui.cmd
```

macOS/Linux:

```bash
./run_ui.sh
```

Open `http://127.0.0.1:7860` in a browser.

## 3. Open Inference

Click `Inference`.

You can either:

- select `datasets/demo/hard_ore_demo.jpg`;
- select `datasets/demo/normal_ore_demo.jpg`;
- upload your own OM image with the upload control.

## 4. Choose Model And Device

Model options:

- `binary segmentation`: predicts the trusted ore/background outline.
- `finetuned CT-UNet ore/talc segmentation`: predicts background, ore, and talc with the finetuned CT-UNet checkpoint at `models/source_binary_segmentation_ct_unet/001/best.pt`.
- `ore segmentation`: predicts mineral class labels directly.

For CPU-only machines:

- set `Device` to `auto` or `cpu`;
- use `Batch size = 1` on low-RAM machines;
- defaults are `Tile size = 512`, `Overlap = 0`, and `Batch size = 16`.

Click `Run prediction`.

## 5. Review Results

After the job completes, the viewer shows:

- raw image;
- raw image plus mask overlay;
- mask only.

Use `+ scale`, `- scale`, `Crop area`, and `Full image` to inspect the same region across all three panels. The metrics table reports class pixel counts and fractions for the full image and the visible crop.

## 6. Optional Intergrowth Review

Select `normal ore intergrowth mask` or `hard ore intergrowth mask` after a completed prediction. The UI automatically runs intergrowth classification when needed, then shows the selected soft mask. The metric table reports `Normal ore / ore pixels`.

## 7. Output Location

Inference outputs are saved under:

```text
data_work/predictions/ui/panorama/<job_id>/
```

Important files include:

- `metadata.json`;
- `ore_mask.png` for binary segmentation;
- `ore_probability.png` and `ore_confidence.png` for binary segmentation;
- `ore_multiclass_mask.png` for ore segmentation;
- preview and overlay PNG/JPEG files;
- intergrowth files when an intergrowth mask layer is selected or `Run intergrowth classification` is used.
