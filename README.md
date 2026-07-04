# Ore Detection

Local UI for automated ore segmentation and intergrowth review from reflected-light optical microscopy images of polished sections.

The repository is prepared for CPU-only inference. A user can clone the repo, run the setup script, launch the UI, and predict masks for the included demo images or their own uploaded OM images. No GPU is required.

## What Is Included

- Local stdlib HTTP UI with inference and active-learning review pages.
- Trained source binary ore/background checkpoint:
  `models/source_binary_segmentation/001/best.pt`.
- Trained source multiclass ore checkpoint:
  `models/source_ore_segmentation/001/best.pt`.
- Local morphology intergrowth classifier config:
  `models/intergrowth_classifier/001/classifier.json`.
- Two small demo images:
  `datasets/demo/hard_ore_demo.jpg` and `datasets/demo/normal_ore_demo.jpg`.

Large local datasets and generated outputs are intentionally not part of the runnable repo. User predictions are written under `data_work/`.

## Requirements

- Python 3.13.
- Internet access for first-time dependency installation.
- CPU is enough for inference. GPU is optional and not required by the default setup.

## Setup

Windows:

```bat
setup_ui_cpu.cmd
```

macOS/Linux:

```bash
./setup_ui_cpu.sh
```

The setup script creates `.venv`, installs runtime dependencies from `requirements.txt`, and installs the CPU PyTorch wheel. The CPU install follows the official PyTorch local-install guidance: <https://pytorch.org/get-started/locally/>.

## Run The Inference UI

Windows:

```bat
run_ui.cmd
```

macOS/Linux:

```bash
./run_ui.sh
```

Open:

```text
http://127.0.0.1:7860
```

Detailed inference instructions are in [docs/ui_inference.md](docs/ui_inference.md).

## Verification

After setup, run:

```bat
py -3.13 scripts\verify.py
```

On POSIX systems with the virtualenv active:

```bash
python scripts/verify.py
```

## Optional Development Environments

- `requirements-dev.txt`: tests and notebook support.
- `requirements-training-cuda.txt`: optional CUDA training/notebook environment for machines with compatible NVIDIA GPUs.

The default UI path does not require CUDA.
