# Pretrained Checkpoint

## File

- **checkpoint_best.pt** (~85MB)

## Description

Pretrained DiffPlace model on synthetic placement data.

## Usage

### Fine-tuning on ISPD benchmarks

```bash
python train.py \
  --config engine/conf/finetune.yaml \
  --pretrained_path checkpoints/pretrain/checkpoint_best.pt \
  --data_dir data/ispd2005 \
  --dataset_type ispd
```

### Inference

```bash
python scripts/deploy.py \
  --checkpoint checkpoints/pretrain/checkpoint_best.pt \
  --benchmarks adaptec1 \
  --visualize
```

## Model Architecture

- **Backbone**: VectorGNNV2Global
- **Parameters**: ~20M
- **Input**: Graph with macro sizes + connectivity
- **Output**: Macro positions (x, y) + rotation

## Training Details

- **Dataset**: Synthetic placement data
- **Objective**: Denoising diffusion loss
- **Optimizer**: AdamW
