#!/usr/bin/env python
"""
DiffPlace Pre-training Script

Usage:
    python train.py --config engine/conf/pretrain.yaml --data_dir <YOUR_DATA_PATH>
    
    # With logging to file:
    python train.py --config engine/conf/pretrain.yaml 2>&1 | tee training.log
"""

import os
import sys
import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

# Add diffusion to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.diffplace import DiffPlace
from engine.training.training_utils import (
    AMPTrainer, CosineAnnealingWithWarmup, EMAModel, 
    TransferLearningManager, smart_load_weights
)
from engine.datasets.synthetic_dataset import create_dataloader
from engine.datasets.ispd_dataset import create_ispd_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="DiffPlace Training")
    parser.add_argument("--config", type=str, default="engine/conf/pretrain.yaml", help="Config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory (e.g., path/to/synthetic/dataset)")
    parser.add_argument("--dataset_type", type=str, default="synthetic", choices=["synthetic", "ispd"], help="Dataset type")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (includes optimizer)")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Load pretrained weights only (for transfer learning)")
    parser.add_argument("--freeze_encoder_steps", type=int, default=0, help="Freeze backbone for N steps")
    parser.add_argument("--rotation_loss_weight", type=float, default=None, help="Override rotation loss weight")
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--debug", action="store_true", help="Debug mode (smaller dataset)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: str) -> DiffPlace:
    """Create model from config."""
    model_cfg = config.get('model', {})
    guidance_cfg = config.get('guidance', {})
    diffusion_cfg = config.get('diffusion', {})
    
    model = DiffPlace(
        hidden_size=model_cfg.get('hidden_size', 256),
        num_blocks=model_cfg.get('num_blocks', 8),
        layers_per_block=model_cfg.get('layers_per_block', 2),
        num_heads=model_cfg.get('num_heads', 8),
        num_rotations=model_cfg.get('num_rotations', 4),
        rotation_temperature=model_cfg.get('rotation_temperature', 1.0),
        global_context_every=model_cfg.get('global_context_every', 2),
        mask_key=model_cfg.get('mask_key', 'is_ports'),
        max_diffusion_steps=diffusion_cfg.get('max_steps', 1000),
        guidance_grid_size=guidance_cfg.get('grid_size', 64),
        density_sigma=guidance_cfg.get('density_sigma', 2.0),
        target_density=guidance_cfg.get('target_density', 1.0),
        hpwl_gamma=guidance_cfg.get('hpwl_gamma', 10.0),
        rudy_threshold=guidance_cfg.get('rudy_threshold', 1.0),
        density_weight=guidance_cfg.get('density_weight', 1.0),
        hpwl_weight=guidance_cfg.get('hpwl_weight', 1.0),
        rudy_weight=guidance_cfg.get('rudy_weight', 0.5),
        guidance_strength=guidance_cfg.get('guidance_strength', 1.0),
        gradient_checkpointing=model_cfg.get('gradient_checkpointing', True),
        device=device,
    ).to(device)
    
    # Enable gradient checkpointing
    if model_cfg.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
    
    return model


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    training_cfg = config.get('training', {})
    
    # Override from args
    if args.steps:
        training_cfg['total_steps'] = args.steps
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print("=" * 70)
    print("DiffPlace Pre-training")
    print("=" * 70)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Output: {output_dir}")
    print(f"Config: {args.config}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get('lr', 1e-4)),
        weight_decay=float(training_cfg.get('weight_decay', 0.01)),
    )
    
    # Create trainer
    trainer = AMPTrainer(
        model=model,
        optimizer=optimizer,
        precision=training_cfg.get('precision', 'bf16'),
        grad_clip_norm=training_cfg.get('grad_clip_norm', 1.0),
        accumulation_steps=training_cfg.get('gradient_accumulation_steps', 4),
    )
    
    # Create scheduler
    total_steps = int(training_cfg.get('total_steps', 100000))
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_steps=int(training_cfg.get('warmup_steps', 2000)),
        total_steps=total_steps,
        min_lr=float(training_cfg.get('min_lr', 1e-6)),
    )
    
    # Create EMA
    ema = None
    if training_cfg.get('use_ema', True):
        ema = EMAModel(model, decay=training_cfg.get('ema_decay', 0.9999))
    
    # Create dataloader
    data_cfg = config.get('data', {})
    print("\nLoading dataset...")
    
    if args.dataset_type == "ispd":
        # ISPD dataset
        loader = create_ispd_dataloader(
            benchmark_dir=args.data_dir,
            benchmarks=data_cfg.get('benchmarks', None),
            batch_size=data_cfg.get('batch_size', 1),
            max_nodes=data_cfg.get('max_nodes', 50000),
            num_workers=0,  # ISPD parsing is slow, avoid multi-worker
        )
    else:
        # Synthetic dataset
        loader = create_dataloader(
            data_dir=args.data_dir,
            split="train",
            batch_size=data_cfg.get('batch_size', 2),
            num_workers=data_cfg.get('num_workers', 4),
            max_samples=100 if args.debug else None,
        )
    
    print(f"  Samples per epoch: {len(loader.dataset)}")
    print(f"  Batches per epoch: {len(loader)}")
    
    # Training state
    start_step = 0
    best_loss = float('inf')
    
    # Transfer Learning: Load pretrained weights (reset optimizer)
    transfer_manager = None
    if args.pretrained_path:
        print(f"\nLoading pretrained weights: {args.pretrained_path}")
        smart_load_weights(model, args.pretrained_path, verbose=True)
        
        # Get freeze steps from args or config
        freeze_steps = args.freeze_encoder_steps
        if freeze_steps == 0:
            transfer_cfg = config.get('transfer', {})
            freeze_steps = int(transfer_cfg.get('freeze_encoder_steps', 0))
        
        if freeze_steps > 0:
            transfer_manager = TransferLearningManager(
                model, 
                freeze_steps=freeze_steps,
                pretrained_path=None,  # Already loaded
                verbose=True,
            )
            print(f"  Freeze backbone for {freeze_steps} steps")
    
    # Resume: Load full checkpoint (includes optimizer)
    elif args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"  Resumed at step {start_step}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    log_every = training_cfg.get('log_every', 50)
    save_every = training_cfg.get('save_every', 5000)
    
    step = start_step
    epoch = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    log_count = 0
    t_start = time.time()
    
    while step < total_steps:
        epoch += 1
        
        for batch in loader:
            if step >= total_steps:
                break
            
            # Move to device
            batch = batch.to(device)
            
            # Transfer learning: update freeze status
            if transfer_manager:
                transfer_manager.step(step)
            
            # Get rotation loss weight
            rot_weight = args.rotation_loss_weight
            if rot_weight is None:
                rot_weight = float(training_cfg.get('rotation_loss_weight', 0.5))
            
            # Training step
            losses = trainer.training_step(
                x_0=batch.pos.unsqueeze(0) if batch.pos.dim() == 2 else batch.pos,
                rot_true=batch.rot_label.unsqueeze(0) if batch.rot_label.dim() == 1 else batch.rot_label,
                cond=batch,
                rotation_loss_weight=rot_weight,
            )
            
            # Update scheduler and EMA
            scheduler.step()
            if ema:
                ema.update()
            
            # Accumulate stats
            running_loss += losses['total']
            running_grad_norm += losses['grad_norm']
            log_count += 1
            step += 1
            
            # Log
            if step % log_every == 0:
                avg_loss = running_loss / log_count
                avg_grad_norm = running_grad_norm / log_count
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                
                print(f"[Step {step:6d}/{total_steps}] "
                      f"loss={avg_loss:.4f} "
                      f"grad={avg_grad_norm:.3f} "
                      f"lr={lr:.2e} "
                      f"mem={gpu_mem:.1f}GB "
                      f"eta={eta/60:.0f}min")
                
                running_loss = 0.0
                running_grad_norm = 0.0
                log_count = 0
            
            # Save checkpoint
            if step % save_every == 0 or step == total_steps:
                checkpoint_path = output_dir / f"checkpoint_{step:06d}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'config': config,
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
                
                # Save best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = output_dir / "checkpoint_best.pt"
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'best_loss': best_loss,
                    }, best_path)
                    print(f"  New best model: loss={best_loss:.4f}")
    
    # Final save
    final_path = output_dir / "checkpoint_final.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Training Complete!")
    print(f"  Total steps: {step}")
    print(f"  Total time: {elapsed/3600:.1f} hours")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved: {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
