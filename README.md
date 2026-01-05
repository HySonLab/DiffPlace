# DiffPlace - Diffusion-based Macro Placement

Physics-informed diffusion model for VLSI macro placement with analytical legalization.

## Project Structure

```
diffplace/
├── train.py                 # Training script
├── scripts/
│   └── deploy.py            # Inference
├── engine/
│   ├── diffplace.py         # Main model (DiffPlace)
│   ├── models.py            # Legacy models
│   ├── utils.py
│   ├── diffusion/           # Diffusion components
│   ├── training/            # Training utilities
│   ├── networks/            # Neural networks
│   ├── datasets/            # Data loading
│   └── conf/                # Configs
└── data/
    └── ispd2005/            # Benchmarks (see README inside)
```

## Quick Start

### Training
```bash
python train.py --config engine/conf/pretrain.yaml --data_dir path/to/synthetic/data
python train.py --config engine/conf/finetune.yaml --dataset_type ispd --data_dir data/ispd2005
```

### Inference
```bash
python scripts/deploy.py \
  --checkpoint path/to/checkpoint.pt \
  --benchmarks adaptec1 --visualize
```

## Features

- **Diffusion placement**: DDIM with density guidance
- **Physics legalization**: Momentum optimizer with cooling
- **Zero overlap**: OccupancyGrid finisher

## Requirements

```bash
conda activate diffplace 
```
