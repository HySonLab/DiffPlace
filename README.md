# DiffPlace - A Conditional Diffusion Framework for Simultaneous VLSI Placement Beyond Sequential Paradigms

Paper: [![arXiv](https://img.shields.io/badge/arXiv-2510.15897-b31b1b.svg)](https://arxiv.org/pdf/2510.15897)

## Experimental Results

Below are the placement results on ISPD2005 benchmarks using DiffPlace.

| Adaptec1 | Adaptec2 | Adaptec3 | Adaptec4 |
|----------|----------|----------|----------|
| ![adaptec1](asset/images/adaptec1_mixed.png) | ![adaptec2](asset/images/adaptec2_mixed.png) | ![adaptec3](asset/images/adaptec3_mixed.png) | ![adaptec4](asset/images/adaptec4_mixed.png) |

| BigBlue1 | BigBlue2 | BigBlue3 | BigBlue4 |
|----------|----------|----------|----------|
| ![bigblue1](asset/images/bigblue1_mixed.png) | ![bigblue2](asset/images/bigblue2_mixed.png) | ![bigblue3](asset/images/bigblue3_mixed.png) | ![bigblue4](asset/images/bigblue4_mixed.png) |

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
python scripts/deploy_nangate45.py \
  --checkpoint path/to/checkpoint.pt \
  --design_dir path/to/MacroPlacement/Flows/NanGate45/ariane133 \
  --output_pkl results/ariane133.pkl
```

## Data

Download ISPD2005 benchmarks from [DREAMPlace](https://github.com/limbo018/DREAMPlace/tree/master/benchmarks)

Additional modern SoC benchmarks (Ariane, NVDLA, MemPool, BlackParrot) available from:
- [TILOS MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement) - Macro placement benchmarks and flows
- [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) - Open-source RTL-to-GDS flow

## Requirements

```bash
pip install -r requirements.txt
```

