"""
DiffPlace Training Utilities

Performance optimizations:
- Mixed Precision Training (AMP)
- Gradient Clipping
- Learning Rate Scheduling
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
import math


class AMPTrainer:
    """
    Automatic Mixed Precision (AMP) training wrapper.
    
    Reduces VRAM usage by ~50% and speeds up matrix operations
    by using FP16/BF16 for forward pass and FP32 for gradients.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        precision: str = "fp16",  # "fp16", "bf16", or "fp32"
        grad_clip_norm: float = 1.0,
        grad_clip_value: Optional[float] = None,
        accumulation_steps: int = 1,
    ):
        """
        Args:
            model: DiffPlace model
            optimizer: Optimizer instance
            precision: "fp16", "bf16", or "fp32"
            grad_clip_norm: Max gradient norm (0 = disabled)
            grad_clip_value: Max gradient value (None = disabled)
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.optimizer = optimizer
        self.precision = precision
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
        # Initialize scaler for fp16
        self.scaler = None
        if precision == "fp16" and torch.cuda.is_available():
            self.scaler = GradScaler()
        
        # Determine autocast dtype
        self.use_amp = precision in ["fp16", "bf16"]
        if precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32
    
    def training_step(
        self,
        x_0: torch.Tensor,
        rot_true: torch.Tensor,
        cond,
        rotation_loss_weight: float = 0.5,
    ) -> Dict[str, float]:
        """
        Single training step with AMP and gradient clipping.
        
        Args:
            x_0: Clean positions (B, V, 2)
            rot_true: True rotation indices (B, V)
            cond: Conditioning data
            rotation_loss_weight: Weight for rotation loss
            
        Returns:
            Dict with loss values
        """
        self.model.train()
        
        # Forward pass with autocast
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            losses = self.model.loss(x_0, rot_true, cond, rotation_loss_weight)
            loss = losses['total'] / self.accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.step_count += 1
        
        # Optimizer step with gradient clipping
        if self.step_count % self.accumulation_steps == 0:
            # Unscale gradients for clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            if self.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.grad_clip_norm
                )
            else:
                grad_norm = self._compute_grad_norm()
            
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.grad_clip_value
                )
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        else:
            grad_norm = 0.0
        
        return {
            'total': losses['total'].item(),
            'position': losses['position'].item(),
            'rotation': losses['rotation'].item(),
            'grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
        }
    
    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm)


class CosineAnnealingWithWarmup:
    """
    Cosine annealing learning rate schedule with linear warmup.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr * base_lr / self.base_lrs[0]
    
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of parameters that is updated with EMA.
    Often improves sample quality.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# TRANSFER LEARNING UTILITIES
# ============================================================================

def smart_load_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Flexibly load weights from checkpoint with shape mismatch handling.
    
    This is essential for transfer learning where:
    - Pre-train on simple Pickle dataset
    - Fine-tune on complex ISPD dataset
    
    Logic:
    1. Load state_dict from checkpoint
    2. For each layer: if shapes match -> load, else -> skip & warn
    3. Layers with shape mismatch keep their random initialization
    
    Args:
        model: Target model (DiffPlace)
        checkpoint_path: Path to .pt or .pth checkpoint
        strict: If True, raise error on mismatch (default: False)
        verbose: Print loading info
        
    Returns:
        Dict with 'loaded', 'skipped', 'missing' layer lists
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        source_state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        source_state = checkpoint['state_dict']
    else:
        source_state = checkpoint
    
    target_state = model.state_dict()
    
    loaded = []
    skipped = []
    missing = []
    
    for name, target_param in target_state.items():
        if name in source_state:
            source_param = source_state[name]
            
            if source_param.shape == target_param.shape:
                # Shape matches - load
                target_state[name] = source_param
                loaded.append(name)
            else:
                # Shape mismatch - skip
                skipped.append(f"{name}: source={list(source_param.shape)} vs target={list(target_param.shape)}")
                if strict:
                    raise RuntimeError(f"Shape mismatch for {name}")
        else:
            missing.append(name)
    
    # Load the modified state dict
    model.load_state_dict(target_state, strict=False)
    
    if verbose:
        print(f"\n=== Smart Weight Loading ===")
        print(f"  Loaded: {len(loaded)} layers")
        print(f"  Skipped (shape mismatch): {len(skipped)} layers")
        print(f"  Missing (new layers): {len(missing)} layers")
        
        if skipped:
            print(f"\n  ⚠ Shape mismatches (randomly initialized):")
            for s in skipped[:5]:  # Show first 5
                print(f"    - {s}")
            if len(skipped) > 5:
                print(f"    ... and {len(skipped) - 5} more")
    
    return {
        'loaded': loaded,
        'skipped': skipped,
        'missing': missing,
    }


def freeze_backbone(model: nn.Module, verbose: bool = True) -> int:
    """
    Freeze backbone (GNN blocks), keep heads trainable.
    
    Freezes:
    - backbone.blocks (VectorGNNBlocks)
    - backbone.global_contexts
    - backbone.input_proj
    - backbone.pos_encoder
    
    Keeps trainable:
    - backbone.position_head
    - backbone.rotation_head
    - guidance module
    
    Returns:
        Number of frozen parameters
    """
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # Freeze backbone blocks
        should_freeze = any(pattern in name for pattern in [
            'backbone.blocks',
            'backbone.global_contexts', 
            'backbone.input_proj',
            'backbone.pos_encoder',
            'backbone.cond_encoder',
            'backbone.time_cond',
        ])
        
        # Keep heads trainable
        should_keep = any(pattern in name for pattern in [
            'position_head',
            'rotation_head',
            'guidance',
        ])
        
        if should_freeze and not should_keep:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            trainable_count += param.numel()
    
    if verbose:
        total = frozen_count + trainable_count
        print(f"\n=== Backbone Frozen ===")
        print(f"  Frozen: {frozen_count:,} params ({100*frozen_count/total:.1f}%)")
        print(f"  Trainable: {trainable_count:,} params ({100*trainable_count/total:.1f}%)")
    
    return frozen_count


def unfreeze_backbone(model: nn.Module, verbose: bool = True) -> int:
    """
    Unfreeze all parameters.
    
    Returns:
        Number of unfrozen parameters
    """
    unfrozen = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen += param.numel()
    
    if verbose:
        print(f"\n=== Backbone Unfrozen ===")
        print(f"  Unfrozen: {unfrozen:,} params")
    
    return unfrozen


class TransferLearningManager:
    """
    Manages transfer learning workflow with freeze scheduling.
    
    Usage:
        manager = TransferLearningManager(model, freeze_steps=1000)
        
        for step in range(total_steps):
            manager.step(step)  # Auto freeze/unfreeze
            losses = trainer.training_step(...)
    """
    
    def __init__(
        self,
        model: nn.Module,
        freeze_steps: int = 0,
        pretrained_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model: DiffPlace model
            freeze_steps: Number of steps to keep backbone frozen (0 = no freeze)
            pretrained_path: Path to pretrained checkpoint
            verbose: Print status messages
        """
        self.model = model
        self.freeze_steps = freeze_steps
        self.verbose = verbose
        self.is_frozen = False
        self.current_step = 0
        
        # Load pretrained weights
        if pretrained_path:
            self.load_result = smart_load_weights(model, pretrained_path, verbose=verbose)
        else:
            self.load_result = None
        
        # Initial freeze if needed
        if freeze_steps > 0:
            freeze_backbone(model, verbose=verbose)
            self.is_frozen = True
    
    def step(self, current_step: int):
        """Update freeze status based on current step."""
        self.current_step = current_step
        
        # Unfreeze after freeze_steps
        if self.is_frozen and current_step >= self.freeze_steps:
            unfreeze_backbone(self.model, verbose=self.verbose)
            self.is_frozen = False
            if self.verbose:
                print(f"  (at step {current_step})")
    
    def get_status(self) -> Dict:
        """Get current status."""
        return {
            'is_frozen': self.is_frozen,
            'freeze_steps': self.freeze_steps,
            'current_step': self.current_step,
            'load_result': self.load_result,
        }


# ============================================================================
# QUICK TRAINING LOOP EXAMPLE
# ============================================================================

def train_step_simple(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_0: torch.Tensor,
    rot_true: torch.Tensor,
    cond,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    scaler: GradScaler = None,
) -> Dict[str, float]:
    """
    Simple training step with AMP and gradient clipping.
    
    Usage:
        scaler = GradScaler() if use_amp else None
        
        for batch in dataloader:
            losses = train_step_simple(model, optimizer, x_0, rot, cond, 
                                       use_amp=True, scaler=scaler)
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward with AMP
    with autocast(enabled=use_amp):
        losses = model.loss(x_0, rot_true, cond)
        loss = losses['total']
    
    # Backward
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    
    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    
    return {
        'total': loss.item(),
        'position': losses['position'].item(),
        'rotation': losses['rotation'].item(),
        'grad_norm': grad_norm.item(),
    }
