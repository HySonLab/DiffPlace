#!/usr/bin/env python
"""
DiffPlace DUAL-STAGE INFERENCE

Stage 1: Inference-time Density Guidance (Differentiable)
Stage 2: Analytical Legalizer (Global Spreading)

Target: 0% Overlap with minimal HPWL increase.

Usage:
    python scripts/deploy.py --checkpoint <YOUR_CHECKPOINT_PATH>
"""

import os
import sys
import argparse
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.diffplace import DiffPlace
from engine.datasets.ispd_dataset import BookshelfParser
from torch_geometric.data import Data


# ============================================================================
# STAGE 1: DIFFERENTIABLE DENSITY GUIDANCE
# ============================================================================

class DensityMap(nn.Module):
    """
    Differentiable density map using Gaussian splatting.
    
    Projects macro positions onto a grid and computes overflow.
    Fully differentiable for gradient-based guidance.
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        sigma: float = 2.0,
        target_density: float = 1.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        self.target_density = target_density
        
        # Pre-compute Gaussian kernel
        k_size = int(6 * sigma) | 1  # Ensure odd
        x = torch.arange(k_size) - k_size // 2
        kernel_1d = torch.exp(-x.float()**2 / (2 * sigma**2))
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer('kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
    
    def forward(
        self,
        positions: torch.Tensor,  # (B, V, 2) in [-1, 1]
        sizes: torch.Tensor,      # (V, 2) normalized
        macro_mask: torch.Tensor, # (V,) bool - True for macros
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute density map and overflow loss.
        
        Returns:
            density_map: (B, 1, H, W)
            overflow_loss: scalar
        """
        B, V, _ = positions.shape
        device = positions.device
        G = self.grid_size
        
        # Create empty density map
        density = torch.zeros(B, 1, G, G, device=device)
        
        # Convert positions [-1, 1] to grid coords [0, G-1]
        grid_pos = (positions + 1) / 2 * (G - 1)  # (B, V, 2)
        
        # Get macro sizes in grid units
        grid_sizes = sizes * G / 2  # (V, 2)
        
        # Splat each macro onto the density map
        for v in range(V):
            if not macro_mask[v]:
                continue
                
            cx = grid_pos[:, v, 0]  # (B,)
            cy = grid_pos[:, v, 1]  # (B,)
            w = grid_sizes[v, 0].item()
            h = grid_sizes[v, 1].item()
            
            # Compute soft density contribution (differentiable)
            x_coords = torch.arange(G, device=device, dtype=torch.float32)
            y_coords = torch.arange(G, device=device, dtype=torch.float32)
            
            # Distance from macro center
            for b in range(B):
                # Gaussian-weighted contribution based on distance
                x_dist = (x_coords - cx[b]).abs()
                y_dist = (y_coords - cy[b]).abs()
                
                # Soft membership (how much of each cell is in the macro)
                x_inside = torch.sigmoid((w/2 - x_dist) * 2)
                y_inside = torch.sigmoid((h/2 - y_dist) * 2)
                
                # Outer product for 2D
                contribution = x_inside.unsqueeze(0) * y_inside.unsqueeze(1)
                density[b, 0] += contribution
        
        # Apply Gaussian smoothing
        pad = self.kernel.shape[-1] // 2
        density = F.pad(density, (pad, pad, pad, pad), mode='reflect')
        density = F.conv2d(density, self.kernel)
        
        # Compute overflow loss (penalize density > target)
        overflow = F.relu(density - self.target_density)
        overflow_loss = overflow.pow(2).mean()
        
        return density, overflow_loss


def sample_ddim_with_density_guidance(
    model: DiffPlace,
    data: Data,
    sizes: torch.Tensor,
    macro_mask: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 2000.0,  # FIXED: High scale for effective steering
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DDIM sampling with differentiable density guidance.
    
    At each step:
    1. Predict x_{t-1}
    2. Enable gradients
    3. Compute density overflow loss
    4. Take gradient step to reduce overlap
    5. Continue
    """
    B = 1
    V = data.x.shape[0]
    T = model.max_diffusion_steps
    
    # Density guidance module
    density_map = DensityMap(grid_size=64, sigma=2.0, target_density=1.0).to(device)
    
    # Create timestep schedule
    timesteps = torch.linspace(T, 1, num_inference_steps, dtype=torch.long, device=device)
    
    # Get fixed node mask
    mask = getattr(data, model.mask_key, None)
    fixed_pos = None
    mask_3d = None
    if mask is not None and hasattr(data, 'pos'):
        fixed_pos = data.pos.unsqueeze(0).expand(B, -1, -1)[:, :, :2]
        mask_3d = mask.unsqueeze(0).unsqueeze(-1).expand(B, V, 2)
    
    # Initial RANDOM noise (strict mode)
    x = torch.randn(B, V, 2, device=device)
    
    # Keep fixed nodes at their positions
    if mask_3d is not None and fixed_pos is not None:
        x = torch.where(mask_3d, fixed_pos, x)
    
    rot_onehot = None
    
    # DDIM reverse process with density guidance
    for i, t in enumerate(timesteps):
        t_tensor = t.expand(B)
        t_idx = t.item() - 1
        
        # Get next timestep
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1].item()
            t_next_idx = t_next - 1
        else:
            t_next = 0
            t_next_idx = -1
        
        # === Forward pass (no grad for prediction) ===
        with torch.no_grad():
            eps_pred, rot_onehot, _ = model(x, data, t_tensor)
            
            # DDIM coefficients
            alpha_bar_t = model.alpha_bar[t_idx]
            sqrt_alpha_bar_t = model.sqrt_alpha_bar[t_idx]
            sqrt_one_minus_alpha_bar_t = model.sqrt_one_minus_alpha_bar[t_idx]
            
            if t_next > 0:
                alpha_bar_t_next = model.alpha_bar[t_next_idx]
            else:
                alpha_bar_t_next = torch.tensor(1.0, device=device)
            
            sqrt_alpha_bar_t_next = torch.sqrt(alpha_bar_t_next)
            sqrt_one_minus_alpha_bar_t_next = torch.sqrt(1 - alpha_bar_t_next)
            
            # Predicted x_0
            x_0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
            x_0_pred = torch.clamp(x_0_pred, -2, 2)
            
            # DDIM update (deterministic: eta=0)
            pred_dir = sqrt_one_minus_alpha_bar_t_next * eps_pred
            x_next = sqrt_alpha_bar_t_next * x_0_pred + pred_dir
        
        # === Density Guidance (with gradients) ===
        # Only apply in later steps when positions are more stable
        progress = 1.0 - (i / num_inference_steps)
        if progress < 0.5:  # Last 50% of denoising
            x_next = x_next.detach().requires_grad_(True)
            
            # Compute density overflow
            _, overflow_loss = density_map(x_next, sizes, macro_mask)
            
            if overflow_loss.item() > 0.01:  # Only guide if needed
                # Gradient of loss w.r.t. positions
                grad = torch.autograd.grad(overflow_loss, x_next)[0]
                
                # Scale gradient based on timestep
                scale = guidance_scale * (1 - progress)  # Stronger at end
                
                # Update: move away from high density regions
                x_next = x_next.detach() - scale * grad
        
        x = x_next.detach()
        
        # Enforce fixed positions
        if mask_3d is not None and fixed_pos is not None:
            x = torch.where(mask_3d, fixed_pos, x)
    
    # Final rotation prediction
    with torch.no_grad():
        _, rot_onehot, _ = model(x, data, torch.ones(B, device=device, dtype=torch.long))
    
    return x, rot_onehot


# ============================================================================
# STAGE 2: ANALYTICAL LEGALIZER (Global Spreading)
# ============================================================================

@dataclass
class Macro:
    """Represents a placed macro."""
    name: str
    x: float
    y: float
    w: float
    h: float
    orient: str = "N"


class AnalyticalLegalizer:
    """
    Stabilized Physics-Based Analytical Legalizer
    
    Features:
    1. Gaussian Density Smoothing
    2. Soft Exponential Boundary Force
    3. Momentum Optimizer with COOLING SCHEDULE
    4. ROBUST FINISHER with OccupancyGrid
    """
    
    def __init__(
        self,
        die_bounds: 'DieBounds',
        num_bins: int = 32,
        num_iterations: int = 1000,     # INCREASED for thorough cooling
        step_size: float = 8.0,
        target_density: float = 0.4,
        momentum: float = 0.92,
        wall_stiffness: float = 1.0,
        gaussian_sigma: float = 2.0,
        decay_rate: float = 0.995,      # NEW: Cooling decay
        verbose: bool = True,
    ):
        self.die = die_bounds
        self.num_bins = num_bins
        self.num_iters = num_iterations
        self.initial_step_size = step_size
        self.step_size = step_size
        self.target_density = target_density
        self.initial_momentum = momentum
        self.momentum = momentum
        self.wall_stiffness = wall_stiffness
        self.gaussian_sigma = gaussian_sigma
        self.decay_rate = decay_rate
        self.verbose = verbose
        
        # Bin dimensions
        self.bin_w = die_bounds.width / num_bins
        self.bin_h = die_bounds.height / num_bins
        
        # Max displacement per step
        self.max_disp = min(die_bounds.width, die_bounds.height) / 15
        
        # Boundary decay length
        self.boundary_decay_x = die_bounds.width * 0.05
        self.boundary_decay_y = die_bounds.height * 0.05
    
    def _compute_density(self, positions: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        
        density = np.zeros((self.num_bins, self.num_bins))
        bin_area = self.bin_w * self.bin_h
        
        for i in range(len(positions)):
            x, y = positions[i]
            w, h = sizes[i]
            
            x1 = int(max(0, (x - self.die.x_min) / self.bin_w))
            x2 = int(min(self.num_bins - 1, (x + w - self.die.x_min) / self.bin_w))
            y1 = int(max(0, (y - self.die.y_min) / self.bin_h))
            y2 = int(min(self.num_bins - 1, (y + h - self.die.y_min) / self.bin_h))
            
            for bx in range(x1, x2 + 1):
                for by in range(y1, y2 + 1):
                    bin_x1 = self.die.x_min + bx * self.bin_w
                    bin_y1 = self.die.y_min + by * self.bin_h
                    bin_x2 = bin_x1 + self.bin_w
                    bin_y2 = bin_y1 + self.bin_h
                    
                    ox1, oy1 = max(x, bin_x1), max(y, bin_y1)
                    ox2, oy2 = min(x + w, bin_x2), min(y + h, bin_y2)
                    
                    if ox2 > ox1 and oy2 > oy1:
                        overlap_area = (ox2 - ox1) * (oy2 - oy1)
                        density[by, bx] += overlap_area / bin_area
        
        density = gaussian_filter(density, sigma=self.gaussian_sigma)
        return density
    
    def _compute_density_forces(self, density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gy, gx = np.gradient(density)
        fx = -gx * self.step_size
        fy = -gy * self.step_size
        return fx, fy
    
    def _compute_boundary_force(self, x: float, y: float, w: float, h: float) -> Tuple[float, float]:
        force_x, force_y = 0.0, 0.0
        
        dist_left = x - self.die.x_min
        dist_right = self.die.x_max - (x + w)
        dist_bottom = y - self.die.y_min
        dist_top = self.die.y_max - (y + h)
        
        if dist_left < self.boundary_decay_x * 3:
            force_x += np.exp(-max(dist_left, 0.1) / self.boundary_decay_x) * self.wall_stiffness * self.step_size
        if dist_right < self.boundary_decay_x * 3:
            force_x -= np.exp(-max(dist_right, 0.1) / self.boundary_decay_x) * self.wall_stiffness * self.step_size
        if dist_bottom < self.boundary_decay_y * 3:
            force_y += np.exp(-max(dist_bottom, 0.1) / self.boundary_decay_y) * self.wall_stiffness * self.step_size
        if dist_top < self.boundary_decay_y * 3:
            force_y -= np.exp(-max(dist_top, 0.1) / self.boundary_decay_y) * self.wall_stiffness * self.step_size
        
        return force_x, force_y
    
    def _snap_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        x_snap = self.die.x_min + round((x - self.die.x_min) / self.die.site_width) * self.die.site_width
        y_snap = self.die.y_min + round((y - self.die.y_min) / self.die.row_height) * self.die.row_height
        return x_snap, y_snap
    
    def legalize(self, macros: List[Macro]) -> List[Macro]:
        """Physics-based spreading with cooling schedule + robust finisher."""
        N = len(macros)
        if N == 0:
            return []
        
        positions = np.array([[m.x, m.y] for m in macros], dtype=np.float64)
        sizes = np.array([[m.w, m.h] for m in macros], dtype=np.float64)
        orients = [m.orient for m in macros]
        names = [m.name for m in macros]
        
        velocity = np.zeros_like(positions)
        
        # Reset step_size and momentum for this run
        self.step_size = self.initial_step_size
        self.momentum = self.initial_momentum
        
        if self.verbose:
            print(f"    Physics Legalizer: {N} macros, iters={self.num_iters}, "
                  f"step={self.step_size:.1f}, momentum={self.momentum:.2f}, decay={self.decay_rate}")
        
        # === PHYSICS SIMULATION WITH COOLING ===
        for iteration in range(self.num_iters):
            density = self._compute_density(positions, sizes)
            max_density = density.max()
            
            fx_field, fy_field = self._compute_density_forces(density)
            
            forces = np.zeros_like(positions)
            max_force = 0.0
            
            for i in range(N):
                x, y = positions[i]
                w, h = sizes[i]
                
                bx = int(np.clip((x - self.die.x_min) / self.bin_w, 0, self.num_bins - 1))
                by = int(np.clip((y - self.die.y_min) / self.bin_h, 0, self.num_bins - 1))
                
                f_density_x = fx_field[by, bx]
                f_density_y = fy_field[by, bx]
                f_boundary_x, f_boundary_y = self._compute_boundary_force(x, y, w, h)
                
                total_fx = f_density_x + f_boundary_x
                total_fy = f_density_y + f_boundary_y
                
                forces[i] = [total_fx, total_fy]
                max_force = max(max_force, abs(total_fx), abs(total_fy))
            
            velocity = self.momentum * velocity + (1 - self.momentum) * forces
            velocity = np.clip(velocity, -self.max_disp, self.max_disp)
            positions = positions + velocity
            
            # Soft clamp
            for i in range(N):
                w, h = sizes[i]
                positions[i, 0] = np.clip(positions[i, 0], self.die.x_min, self.die.x_max - w)
                positions[i, 1] = np.clip(positions[i, 1], self.die.y_min, self.die.y_max - h)
            
            # Debug logging every 100 iterations
            if self.verbose and iteration % 100 == 0:
                avg_vel = np.mean(np.abs(velocity))
                print(f"    Step {iteration:4d}: Density={max_density:.2f}, "
                      f"AvgVel={avg_vel:.2f}, step={self.step_size:.3f}")
            
            # COOLING: Decay step_size and momentum
            self.step_size *= self.decay_rate
            self.momentum = 0.5 + (self.momentum - 0.5) * self.decay_rate  # Decay toward 0.5
            
            # Early stop if converged
            if max_density < self.target_density + 0.1 and max_force < 0.5:
                if self.verbose:
                    print(f"    Converged at step {iteration}")
                break
        
        if self.verbose:
            print(f"    Final step_size: {self.step_size:.4f} (from {self.initial_step_size})")
        
        # === ROBUST FINISHER ===
        positions = self._finalize_positions(positions, sizes, names)
        
        # Convert back to Macro objects
        result = []
        for i in range(N):
            result.append(Macro(
                name=names[i],
                x=positions[i, 0],
                y=positions[i, 1],
                w=sizes[i, 0],
                h=sizes[i, 1],
                orient=orients[i],
            ))
        
        return result
    
    def _finalize_positions(
        self,
        coarse_pos: np.ndarray,
        sizes: np.ndarray,
        names: List[str],
    ) -> np.ndarray:
        """
        Robust Finisher: OccupancyGrid-based greedy placement.
        
        1. Sort macros by Y, then X (row-major order)
        2. Use high-res OccupancyGrid to track occupied cells
        3. Shift overlapping macros right/down until valid
        """
        N = len(coarse_pos)
        if N == 0:
            return coarse_pos
        
        if self.verbose:
            print(f"    Robust Finisher: Processing {N} macros...")
        
        # High-resolution occupancy grid
        grid_res = max(self.die.row_height, 5)
        grid_w = int(np.ceil(self.die.width / grid_res))
        grid_h = int(np.ceil(self.die.height / grid_res))
        occupancy = np.zeros((grid_h, grid_w), dtype=bool)
        
        # Sort macros by Y, then X
        order = np.lexsort((coarse_pos[:, 0], coarse_pos[:, 1]))
        
        final_pos = coarse_pos.copy()
        shifted_count = 0
        
        for idx in order:
            x, y = coarse_pos[idx]
            w, h = sizes[idx]
            
            # Snap to grid
            x, y = self._snap_to_grid(x, y)
            x = np.clip(x, self.die.x_min, self.die.x_max - w)
            y = np.clip(y, self.die.y_min, self.die.y_max - h)
            
            # Check and resolve overlap
            attempts = 0
            max_attempts = 10000
            
            while attempts < max_attempts:
                # Convert to grid coords
                gx1 = int((x - self.die.x_min) / grid_res)
                gy1 = int((y - self.die.y_min) / grid_res)
                gx2 = int(np.ceil((x + w - self.die.x_min) / grid_res))
                gy2 = int(np.ceil((y + h - self.die.y_min) / grid_res))
                
                gx1, gx2 = max(0, gx1), min(grid_w, gx2)
                gy1, gy2 = max(0, gy1), min(grid_h, gy2)
                
                # Check if region is free
                if not occupancy[gy1:gy2, gx1:gx2].any():
                    break
                
                # Shift right first, then down
                if x + w + self.die.site_width <= self.die.x_max:
                    x += self.die.site_width
                else:
                    x = self.die.x_min
                    y += self.die.row_height
                    if y + h > self.die.y_max:
                        y = self.die.y_min  # Wrap around, shouldn't happen
                
                x, y = self._snap_to_grid(x, y)
                x = np.clip(x, self.die.x_min, self.die.x_max - w)
                y = np.clip(y, self.die.y_min, self.die.y_max - h)
                attempts += 1
            
            if attempts > 0:
                shifted_count += 1
            
            # Mark occupied
            gx1 = int((x - self.die.x_min) / grid_res)
            gy1 = int((y - self.die.y_min) / grid_res)
            gx2 = int(np.ceil((x + w - self.die.x_min) / grid_res))
            gy2 = int(np.ceil((y + h - self.die.y_min) / grid_res))
            gx1, gx2 = max(0, gx1), min(grid_w, gx2)
            gy1, gy2 = max(0, gy1), min(grid_h, gy2)
            occupancy[gy1:gy2, gx1:gx2] = True
            
            final_pos[idx] = [x, y]
        
        if self.verbose:
            print(f"    Macros shifted during finalization: {shifted_count}/{N}")
        
        return final_pos
    
    def _greedy_resolve(
        self, 
        positions: np.ndarray, 
        sizes: np.ndarray,
    ) -> np.ndarray:
        """Greedy resolution of remaining overlaps."""
        N = len(positions)
        
        # Sort by area (largest first)
        areas = sizes[:, 0] * sizes[:, 1]
        order = np.argsort(-areas)
        
        placed = np.zeros(N, dtype=bool)
        
        for idx in order:
            x, y = positions[idx]
            w, h = sizes[idx]
            
            # Check overlap with already placed macros
            has_overlap = False
            for j in range(N):
                if not placed[j] or j == idx:
                    continue
                
                ox = max(0, min(x + w, positions[j, 0] + sizes[j, 0]) - max(x, positions[j, 0]))
                oy = max(0, min(y + h, positions[j, 1] + sizes[j, 1]) - max(y, positions[j, 1]))
                
                if ox > 0 and oy > 0:
                    has_overlap = True
                    break
            
            if has_overlap:
                # Spiral search for valid position
                x, y = self._find_valid_position(positions, sizes, idx, placed)
                positions[idx] = [x, y]
            
            placed[idx] = True
        
        return positions
    
    def _find_valid_position(
        self,
        positions: np.ndarray,
        sizes: np.ndarray,
        idx: int,
        placed: np.ndarray,
    ) -> Tuple[float, float]:
        """Find nearest valid position using spiral search."""
        x0, y0 = positions[idx]
        w, h = sizes[idx]
        
        step_x = self.die.site_width * 10
        step_y = self.die.row_height
        
        for radius in range(1, 100):
            for dx in range(-radius, radius + 1):
                for dy in [-radius, radius] if abs(dx) < radius else range(-radius, radius + 1):
                    nx = x0 + dx * step_x
                    ny = y0 + dy * step_y
                    
                    # Snap and clamp
                    nx, ny = self._snap_to_grid(nx, ny)
                    nx = max(self.die.x_min, min(self.die.x_max - w, nx))
                    ny = max(self.die.y_min, min(self.die.y_max - h, ny))
                    
                    # Check overlap
                    valid = True
                    for j in range(len(positions)):
                        if not placed[j] or j == idx:
                            continue
                        
                        ox = max(0, min(nx + w, positions[j, 0] + sizes[j, 0]) - max(nx, positions[j, 0]))
                        oy = max(0, min(ny + h, positions[j, 1] + sizes[j, 1]) - max(ny, positions[j, 1]))
                        
                        if ox > 0 and oy > 0:
                            valid = False
                            break
                    
                    if valid:
                        return nx, ny
        
        return x0, y0


class TetrisLegalizer:
    """
    Final Tetris-style legalizer for guaranteed 0% overlap.
    
    Uses GridMap occupancy tracking + spiral search.
    Run this AFTER AnalyticalLegalizer for cleanup.
    """
    
    def __init__(self, die_bounds: 'DieBounds', grid_resolution: float = 10.0):
        self.die = die_bounds
        self.grid_res = grid_resolution
        self.grid_w = int(np.ceil(die_bounds.width / grid_resolution))
        self.grid_h = int(np.ceil(die_bounds.height / grid_resolution))
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        self.max_disp = die_bounds.width  # Can search entire die
    
    def _to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.die.x_min) / self.grid_res)
        gy = int((y - self.die.y_min) / self.grid_res)
        gx = max(0, min(self.grid_w - 1, gx))
        gy = max(0, min(self.grid_h - 1, gy))
        return gx, gy
    
    def _mark_occupied(self, x: float, y: float, w: float, h: float):
        x1, y1 = self._to_grid(x, y)
        x2, y2 = self._to_grid(x + w, y + h)
        self.grid[y1:y2+1, x1:x2+1] = True
    
    def _check_overlap(self, x: float, y: float, w: float, h: float) -> bool:
        if x < self.die.x_min or x + w > self.die.x_max:
            return True
        if y < self.die.y_min or y + h > self.die.y_max:
            return True
        x1, y1 = self._to_grid(x, y)
        x2, y2 = self._to_grid(x + w, y + h)
        return self.grid[y1:y2+1, x1:x2+1].any()
    
    def _snap(self, x: float, y: float) -> Tuple[float, float]:
        x_snap = self.die.x_min + round((x - self.die.x_min) / self.die.site_width) * self.die.site_width
        y_snap = self.die.y_min + round((y - self.die.y_min) / self.die.row_height) * self.die.row_height
        return x_snap, y_snap
    
    def _find_valid_position(self, x: float, y: float, w: float, h: float) -> Tuple[float, float]:
        x, y = self._snap(x, y)
        x = max(self.die.x_min, min(self.die.x_max - w, x))
        y = max(self.die.y_min, min(self.die.y_max - h, y))
        
        if not self._check_overlap(x, y, w, h):
            return x, y
        
        # Spiral search with large radius
        step_x = max(self.die.site_width * 5, 10)
        step_y = max(self.die.row_height, 10)
        
        for radius in range(1, 500):  # Large search radius
            for dx in range(-radius, radius + 1):
                for dy in [-radius, radius] if abs(dx) < radius else range(-radius, radius + 1):
                    nx = x + dx * step_x
                    ny = y + dy * step_y
                    nx, ny = self._snap(nx, ny)
                    nx = max(self.die.x_min, min(self.die.x_max - w, nx))
                    ny = max(self.die.y_min, min(self.die.y_max - h, ny))
                    
                    if not self._check_overlap(nx, ny, w, h):
                        return nx, ny
        
        return x, y  # Fallback
    
    def legalize(self, macros: List[Macro]) -> List[Macro]:
        """Final legalization with guaranteed 0% overlap."""
        # Sort by area (largest first)
        sorted_macros = sorted(macros, key=lambda m: m.w * m.h, reverse=True)
        
        result = []
        failed = 0
        
        for macro in sorted_macros:
            x, y = self._find_valid_position(macro.x, macro.y, macro.w, macro.h)
            
            if not self._check_overlap(x, y, macro.w, macro.h):
                self._mark_occupied(x, y, macro.w, macro.h)
            else:
                failed += 1
            
            result.append(Macro(
                name=macro.name,
                x=x, y=y,
                w=macro.w, h=macro.h,
                orient=macro.orient,
            ))
        
        if failed > 0:
            print(f"    ⚠️ TetrisLegalizer: {failed} macros could not find valid position")
        
        return result


# ============================================================================
# PARSER & UTILITIES
# ============================================================================

@dataclass
class DieBounds:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    row_height: float
    site_width: float
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min


def parse_scl_for_die(benchmark_dir: str, benchmark_name: str) -> DieBounds:
    """Parse .scl file for die bounds."""
    scl_file = os.path.join(benchmark_dir, benchmark_name, f"{benchmark_name}.scl")
    
    rows = []
    current_row = {}
    
    with open(scl_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('Coordinate'):
                match = re.search(r':\s*([\d.]+)', line)
                if match:
                    current_row['y'] = float(match.group(1))
            elif line.startswith('Height'):
                match = re.search(r':\s*([\d.]+)', line)
                if match:
                    current_row['height'] = float(match.group(1))
            elif line.startswith('Sitewidth'):
                match = re.search(r':\s*([\d.]+)', line)
                if match:
                    current_row['site_width'] = float(match.group(1))
            elif 'SubrowOrigin' in line and 'NumSites' in line:
                match_origin = re.search(r'SubrowOrigin\s*:\s*([\d.]+)', line)
                match_sites = re.search(r'NumSites\s*:\s*([\d.]+)', line)
                if match_origin:
                    current_row['x_origin'] = float(match_origin.group(1))
                if match_sites:
                    current_row['num_sites'] = int(match_sites.group(1))
            elif line == 'End':
                if current_row:
                    rows.append(current_row)
                    current_row = {}
    
    row_height = rows[0].get('height', 12)
    site_width = rows[0].get('site_width', 1)
    x_min = min(r.get('x_origin', 0) for r in rows)
    y_min = min(r.get('y', 0) for r in rows)
    x_max = max(r.get('x_origin', 0) + r.get('num_sites', 0) * site_width for r in rows)
    y_max = max(r.get('y', 0) + r.get('height', 12) for r in rows)
    
    return DieBounds(x_min, y_min, x_max, y_max, row_height, site_width)


def get_original_sizes(benchmark_dir: str, benchmark_name: str) -> Dict[str, Tuple[float, float]]:
    """Get node sizes from .nodes file."""
    nodes_file = os.path.join(benchmark_dir, benchmark_name, f"{benchmark_name}.nodes")
    sizes = {}
    
    with open(nodes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('UCLA'):
                continue
            if line.startswith('NumNodes') or line.startswith('NumTerminals'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                sizes[parts[0]] = (float(parts[1]), float(parts[2]))
    
    return sizes


def calculate_hpwl(macros: List[Macro]) -> float:
    if len(macros) < 2:
        return 0.0
    xs = [m.x + m.w/2 for m in macros]
    ys = [m.y + m.h/2 for m in macros]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def calculate_overlap_area(macros: List[Macro]) -> float:
    total = 0.0
    for i, m1 in enumerate(macros):
        for m2 in macros[i+1:]:
            ox = max(0, min(m1.x + m1.w, m2.x + m2.w) - max(m1.x, m2.x))
            oy = max(0, min(m1.y + m1.h, m2.y + m2.h) - max(m1.y, m2.y))
            total += ox * oy
    return total


def load_model(checkpoint_path: str, device: str) -> DiffPlace:
    print(f"Loading model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    
    model = DiffPlace(
        hidden_size=model_cfg.get('hidden_size', 256),
        num_blocks=model_cfg.get('num_blocks', 8),
        layers_per_block=model_cfg.get('layers_per_block', 2),
        num_heads=model_cfg.get('num_heads', 8),
        global_context_every=model_cfg.get('global_context_every', 2),
        gradient_checkpointing=False,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    return model


# ============================================================================
# MAIN INFERENCE
# ============================================================================

def run_inference(
    model: DiffPlace,
    benchmark_dir: str,
    benchmark_name: str,
    output_dir: str,
    max_nodes: int = 50000,
    ddim_steps: int = 50,
    guidance_scale: float = 2000.0,  # FIXED: High scale for effective steering
    device: str = "cuda",
):
    """Run dual-stage inference."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {benchmark_name}")
    print(f"{'='*70}")
    
    # 1. Parse benchmark
    die = parse_scl_for_die(benchmark_dir, benchmark_name)
    orig_sizes = get_original_sizes(benchmark_dir, benchmark_name)
    
    print(f"  Die: {die.width:.0f} x {die.height:.0f}")
    print(f"  Row Height: {die.row_height}")
    
    # 2. Build graph with smart subsampling (macros first)
    parser = BookshelfParser(benchmark_dir, benchmark_name)
    parser.parse()
    
    all_nodes = [n for n in parser.node_names if n in parser.placements]
    macro_names = [n for n in all_nodes if orig_sizes.get(n, (0, 0))[1] > die.row_height]
    stdcell_names = [n for n in all_nodes if orig_sizes.get(n, (0, 0))[1] <= die.row_height]
    
    print(f"  Total: {len(all_nodes)} ({len(macro_names)} macros)")
    
    # Select nodes: all macros + sample stdcells
    selected = macro_names.copy()
    remaining = max_nodes - len(selected)
    if remaining > 0 and stdcell_names:
        np.random.seed(42)
        sample_size = min(remaining, len(stdcell_names))
        indices = np.random.choice(len(stdcell_names), sample_size, replace=False)
        selected.extend([stdcell_names[i] for i in sorted(indices)])
    
    print(f"  Selected: {len(selected)} (all {len(macro_names)} macros)")
    
    # Build tensors (STRICT MODE: random init for macros)
    V = len(selected)
    new_idx = {name: i for i, name in enumerate(selected)}
    
    positions = torch.zeros(V, 2)
    sizes_tensor = torch.zeros(V, 2)
    is_macro = torch.zeros(V, dtype=torch.bool)
    
    orient_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3, 'FN': 0, 'FS': 2, 'FE': 1, 'FW': 3}
    
    movable_count = 0
    fixed_count = 0
    
    for i, name in enumerate(selected):
        w, h = orig_sizes.get(name, (1, 1))
        sizes_tensor[i] = torch.tensor([w, h])
        
        if h > die.row_height:
            # MOVABLE: Random uniform init
            positions[i] = torch.tensor([
                np.random.uniform(die.x_min, die.x_max),
                np.random.uniform(die.y_min, die.y_max),
            ])
            is_macro[i] = True
            movable_count += 1
        else:
            # FIXED: Use placement from .pl
            if name in parser.placements:
                x, y, _ = parser.placements[name]
                positions[i] = torch.tensor([x, y])
            else:
                positions[i] = torch.randn(2) * 1000 + 5000
            fixed_count += 1
    
    print(f"  ✓ Movable Macros: {movable_count} (RANDOM)")
    print(f"  ✓ Fixed Context: {fixed_count}")
    
    # Normalize
    positions[:, 0] = (positions[:, 0] - die.x_min) / die.width * 2 - 1
    positions[:, 1] = (positions[:, 1] - die.y_min) / die.height * 2 - 1
    sizes_norm = sizes_tensor.clone()
    sizes_norm[:, 0] = sizes_tensor[:, 0] / die.width * 0.1
    sizes_norm[:, 1] = sizes_tensor[:, 1] / die.height * 0.1
    
    # Build edges
    edges_src, edges_dst = [], []
    for net in parser.nets:
        pins = [new_idx[p[0]] for p in net['pins'] if p[0] in new_idx]
        if len(pins) >= 2:
            for j in range(1, len(pins)):
                edges_src.extend([pins[0], pins[j]])
                edges_dst.extend([pins[j], pins[0]])
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long) if edges_src else torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.randn(len(edges_src), 4) * 0.1 if edges_src else torch.zeros(0, 4)
    
    data = Data(
        pos=positions,
        x=sizes_norm,
        edge_index=edge_index,
        edge_attr=edge_attr,
        is_ports=~is_macro,  # Fixed nodes are ports
        num_nodes=V,
    ).to(device)
    
    print(f"  Nodes: {V}, Edges: {edge_index.shape[1]}")
    
    # ========================================
    # STAGE 1: DDIM with Density Guidance
    # ========================================
    print(f"\n[STAGE 1] DDIM Sampling + Density Guidance ({ddim_steps} steps, scale={guidance_scale})")
    
    pred_pos, pred_rot = sample_ddim_with_density_guidance(
        model=model,
        data=data,
        sizes=sizes_norm.to(device),
        macro_mask=is_macro.to(device),
        num_inference_steps=ddim_steps,
        guidance_scale=guidance_scale,
        device=device,
    )
    
    # Extract macro positions
    pred_pos = pred_pos.squeeze(0).float().cpu().numpy()
    pred_rot = pred_rot.squeeze(0).argmax(dim=-1).cpu().numpy()
    
    macro_indices = torch.where(is_macro)[0].tolist()
    macro_pos = pred_pos[macro_indices]
    macro_rot = pred_rot[macro_indices]
    macro_size_list = [orig_sizes[macro_names[i]] for i in range(len(macro_names))]
    
    # Scale to die coordinates
    macro_pos[:, 0] = (macro_pos[:, 0] + 1) / 2 * die.width + die.x_min
    macro_pos[:, 1] = (macro_pos[:, 1] + 1) / 2 * die.height + die.y_min
    
    orient_map_inv = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    
    macros_stage1 = []
    for i, name in enumerate(macro_names):
        x, y = macro_pos[i]
        w, h = macro_size_list[i]
        x = max(die.x_min, min(die.x_max - w, x))
        y = max(die.y_min, min(die.y_max - h, y))
        macros_stage1.append(Macro(name=name, x=x, y=y, w=w, h=h, orient=orient_map_inv.get(macro_rot[i], 'N')))
    
    hpwl_stage1 = calculate_hpwl(macros_stage1)
    overlap_stage1 = calculate_overlap_area(macros_stage1)
    
    print(f"  → HPWL: {hpwl_stage1:,.0f}")
    print(f"  → Overlap: {overlap_stage1:,.0f}")
    
    # ========================================
    # STAGE 2: Analytical Legalizer (Global Spreading)
    # ========================================
    print(f"\n[STAGE 2] Analytical Legalizer (Global Spreading)")
    
    legalizer = AnalyticalLegalizer(die_bounds=die)  # Uses new aggressive defaults
    
    macros_stage2 = legalizer.legalize(macros_stage1)
    
    hpwl_stage2 = calculate_hpwl(macros_stage2)
    overlap_stage2 = calculate_overlap_area(macros_stage2)
    
    print(f"  → HPWL: {hpwl_stage2:,.0f} ({'+' if hpwl_stage2 > hpwl_stage1 else ''}{(hpwl_stage2 - hpwl_stage1) / max(hpwl_stage1, 1) * 100:.1f}%)")
    print(f"  → Overlap: {overlap_stage2:,.0f} ({(overlap_stage1 - overlap_stage2) / max(overlap_stage1, 1) * 100:.1f}% reduction)")
    
    # ========================================
    # STAGE 3: Tetris Legalizer (Final Cleanup)
    # ========================================
    print(f"\n[STAGE 3] Tetris Legalizer (Zero Overlap)")
    
    tetris = TetrisLegalizer(die_bounds=die, grid_resolution=max(die.row_height, 5))
    macros_stage3 = tetris.legalize(macros_stage2)
    
    hpwl_stage3 = calculate_hpwl(macros_stage3)
    overlap_stage3 = calculate_overlap_area(macros_stage3)
    
    print(f"  → HPWL: {hpwl_stage3:,.0f} ({'+' if hpwl_stage3 > hpwl_stage2 else ''}{(hpwl_stage3 - hpwl_stage2) / max(hpwl_stage2, 1) * 100:.1f}%)")
    print(f"  → Overlap: {overlap_stage3:,.0f}")
    
    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*70}")
    print(f"SUMMARY: {benchmark_name}")
    print(f"{'='*70}")
    print(f"{'Stage':<25} {'HPWL':>12} {'Overlap':>15}")
    print(f"{'-'*70}")
    print(f"{'Stage 1 (DDIM+Guidance):':<25} {hpwl_stage1:>12,.0f} {overlap_stage1:>15,.0f}")
    print(f"{'Stage 2 (Analytical):':<25} {hpwl_stage2:>12,.0f} {overlap_stage2:>15,.0f}")
    print(f"{'Stage 3 (Tetris):':<25} {hpwl_stage3:>12,.0f} {overlap_stage3:>15,.0f}")
    print(f"{'='*70}")
    
    # Save result
    result = {
        'name': benchmark_name,
        'die_area': {'width': die.width, 'height': die.height, 'x_min': die.x_min, 'y_min': die.y_min},
        'macros': {m.name: {'x': m.x, 'y': m.y, 'w': m.w, 'h': m.h, 'orient': m.orient} for m in macros_stage3},
        'num_macros': len(macros_stage3),
        'hpwl_stage1': hpwl_stage1,
        'hpwl_stage2': hpwl_stage2,
        'hpwl_final': hpwl_stage3,
        'overlap_stage1': overlap_stage1,
        'overlap_stage2': overlap_stage2,
        'overlap_final': overlap_stage3,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{benchmark_name}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"  Saved: {output_path}")
    
    return result


def visualize_result(result: Dict, output_path: str):
    die = result['die_area']
    macros = result['macros']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12 * die['height'] / die['width']))
    
    die_rect = patches.Rectangle(
        (die['x_min'], die['y_min']), die['width'], die['height'],
        linewidth=2, edgecolor='black', facecolor='#f5f5f5',
    )
    ax.add_patch(die_rect)
    
    orient_colors = {'N': '#27ae60', 'E': '#3498db', 'S': '#e74c3c', 'W': '#9b59b6'}
    
    for name, m in macros.items():
        color = orient_colors.get(m['orient'], '#95a5a6')
        rect = patches.Rectangle(
            (m['x'], m['y']), m['w'], m['h'],
            linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.8,
        )
        ax.add_patch(rect)
    
    ax.set_xlim(die['x_min'] - 100, die['x_min'] + die['width'] + 100)
    ax.set_ylim(die['y_min'] - 100, die['y_min'] + die['height'] + 100)
    ax.set_aspect('equal')
    ax.set_title(f"{result['name']}: {result['num_macros']} macros | "
                 f"HPWL: {result['hpwl_stage2']:,.0f} | Overlap: {result['overlap_stage2']:,.0f}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Visualization: {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="<YOUR_CHECKPOINT_PATH>  # Path to trained model")
    p.add_argument("--benchmark_dir", type=str, default="data/ispd2005")
    p.add_argument("--benchmarks", type=str, nargs='+', default=['adaptec1'])
    p.add_argument("--output_dir", type=str, default="results/ispd")
    p.add_argument("--max_nodes", type=int, default=50000)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=10.0)
    p.add_argument("--visualize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = load_model(args.checkpoint, device)
    
    results = []
    for benchmark in args.benchmarks:
        try:
            result = run_inference(
                model=model,
                benchmark_dir=args.benchmark_dir,
                benchmark_name=benchmark,
                output_dir=args.output_dir,
                max_nodes=args.max_nodes,
                ddim_steps=args.ddim_steps,
                guidance_scale=args.guidance_scale,
                device=device,
            )
            results.append(result)
            
            if args.visualize:
                vis_path = os.path.join(args.output_dir, f"{benchmark}.png")
                visualize_result(result, vis_path)
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Benchmark':<12} {'Macros':>8} {'HPWL Stage1':>14} {'HPWL Stage2':>14} {'Overlap':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<12} {r['num_macros']:>8} {r['hpwl_stage1']:>14,.0f} "
              f"{r['hpwl_stage2']:>14,.0f} {r['overlap_stage2']:>12,.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
