
# torch-warp-ngp - Instant NGP reimplementation - PyTorch accelerated with Warp

This repository contains the code used to reproduce the experiments reported in the accompanying project practical report. The project focuses on reimplementing instant-ngp in PyTorch and also accelerated in NVIDIA Warp.

This README documents: a reproducible setup, exact commands to reproduce the thesis results, dataset instructions, code structure, and contribution guidelines.

**Status:** Designed for full reproducibility. Follow the steps below exactly to reproduce the results reported in the project report.

**Table of Contents**
- Project Overview
- Quickstart: reproduce thesis results
- Environment & Dependencies
- Data: where to get it and layout
- Exact reproduction commands
- Code structure
- Configuration and deterministic runs
- Results, artifacts & verification
- Contributing and contact

## Project Overview

This repo implements the models, training loops, evaluation and rendering code used in the thesis. Key entry points:
- [train.py](train.py): training entry for experiments
- [model.py](model.py): model definitions
- [rendering.py](rendering.py): rendering utilities and evaluation
- [sh.py](sh.py): spherical encodings
- [encoding.py](encoding.py): multiresolution hash encodings
- [config.py](config.py): configuration for project, edit as needed
- [data.py](data.py): dataset loaders and helpers
- [environment.yml](environment.yml): specific environment used
- scripts/: experiment orchestration (bash) for reproducing results

The codebase is modular and organized so experiments are configurable, reproducible, and auditable.

## Quickstart: reproduce thesis results

Follow these commands to set up the environment and reproduce the experiments exactly as reported.

1) Create the conda environment (recommended):

```bash
conda env create -f environment.yml
conda activate torch-warp-ngp
```

If you prefer a venv/pip workflow:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # if you create this file from environment.yml
```

2) Prepare the datasets (see Data section below).

3) Reproduce the main experiments (exact commands are in the "Exact reproduction commands" section).

Note: the provided `environment.yml` pins package versions used to produce the thesis results. Use it verbatim for exact reproduction.

## Environment & Dependencies

- Primary language: Python 3.10+ (match `environment.yml` contents).
- Core packages: PyTorch (CUDA-enabled matching the driver), NumPy, tqdm, imageio, and other utilities pinned in `environment.yml`.

Recommendations:
- Use the `environment.yml` included here to create a deterministic environment.
- Install a PyTorch build matching your CUDA driver. If unsure, follow the official PyTorch installation instructions for your CUDA version.

If you modify package versions, tests may not match thesis numbers.

## Data: where to get it and layout

All experiments expect datasets to be available under a top-level `data/` directory. The repository does not embed large datasets; instead, follow these steps:

- Create the dataset folder (easiest at the repository root).
- For each dataset used in the thesis, create a subfolder with the dataset name and the expected structure.
- If using the scripts in the scripts folder, please insert this dataset folder where needed

The project used synthetic NeRF. The ideal step is to download the dataset from the original sources and place them in a newly created `data/<dataset_name>/`.

## Exact reproduction commands

The repository contains convenience scripts to run full experiment suites. Use them to reproduce the numbers and figures in the thesis. Make sure to edit the dataset paths.

- Reproduce all ablation experiments (as used in the thesis):

```bash
bash scripts/run_ablations.sh
```

- Reproduce dataset ablation sweeps:

```bash
bash scripts/run_dataset_ablation.sh
```

## Command-line arguments

The training and evaluation scripts expose a command-line interface implemented with `argparse`. Below is the parser used in the code and a concise explanation of what each `--` flag means.

Flag explanations:

- `--exp_name <str>`: Name of the experiment run folder. Used to create the output directory under `results/` or the configured output path. Default: `run_baseline`.
- `--data_root <str>`: Path to the dataset root. Point this to the dataset folder (for example `data/nerf-synthetic/`). Defaults to `Config.DATA_ROOT`.
- `--use_warp` (boolean flag): When present, the code will enable the Warp backend for accelerated operations. Omit to run the standard PyTorch implementation.
- `--iterations <int>`: Total number of training iterations (optimization steps) to run. Controls when training stops.
- `--val_interval <int>`: Number of iterations between validation runs. A validation pass will run every `val_interval` training iterations.
- `--batch_size <int>`: Batch size in terms of rays per training step. Larger values increase memory and throughput.
- `--lr <float>`: Initial learning rate for the optimizer.
- `--n_samples <int>`: Number of sample points sampled along each ray for volume rendering.
- `--l` / `--L <int>`: Number of hash grid levels used by the multi-resolution hash encoding. The parser provides `--l` but stores it in `args.L`.
- `--f` / `--F <int>`: Feature dimension per hash grid level. The parser provides `--f` but stores it in `args.F`.
- `--t` / `--T <int>`: Hash table size — controls the number of hashed entries available per level.
- `--aabb_min <x> <y> <z>`: Minimum (x,y,z) coordinates of the axis-aligned bounding box that defines the scene bounds. Example: `--aabb_min -1.5 -1.5 -1.5`.
- `--aabb_max <x> <y> <z>`: Maximum (x,y,z) coordinates of the axis-aligned bounding box. Example: `--aabb_max 1.5 1.5 1.5`.

Examples:

```bash
# Run a baseline experiment using Warp, with a custom name and dataset path
python train.py --exp_name experiment1 --data_root data/nerf-synthetic/lego --use_warp --iterations 200000 \
	--batch_size 4096 --lr 0.01 --n_samples 64 --l 16 --f 2 --t 1048576 \
	--aabb_min -1.5 -1.5 -1.5 --aabb_max 1.5 1.5 1.5
```

## Results, artifacts & verification

- Output artifacts: model checkpoints and rendered outputs are written to the `runs/` directory by default.
- Metrics & logs: training logs (tensorboard or plain text) are saved alongside checkpoints. Re-run the exact commands in the "Exact reproduction commands" section to reproduce the reported numbers.

Verification checklist to ensure faithful reproduction:

1. Confirm environment was created using `environment.yml`.
2. Confirm dataset files are placed under `data/` with the expected layout.
3. Run the exact script from `scripts/` (the script documents hyperparameters and seeds).
4. Match final metric values against the thesis appendix; discrepancies typically indicate environment or data mismatches.

## Troubleshooting

- CUDA/cuDNN mismatch: reinstall a PyTorch wheel matching your system's CUDA.
- Numerical differences: ensure seeds, `cudnn.deterministic`, and environment versions match `environment.yml`.
- Missing data: verify dataset path and run any `scripts/download_*.sh` helper scripts if available.