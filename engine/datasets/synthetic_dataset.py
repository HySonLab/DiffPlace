"""
DiffPlace Synthetic Dataset

Smart dataset with on-the-fly:
- Edge-to-Net conversion for Trinity guidance
- Random rotation injection
- Position normalization
"""

import torch
import pickle
import os
import glob
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DiffPlacePickleDataset(Dataset):
    """
    Dataset for DiffPlace synthetic data stored as pickle files.
    
    Each pickle file contains a list of (x, cond) tuples where:
    - x: (V, 2) ground truth positions
    - cond: PyG Data with node features, edge_index, edge_attr
    
    On-the-fly transformations:
    - Converts edge_index to net structure for Trinity guidance
    - Injects random rotation labels
    - Normalizes positions to [-1, 1]
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train" or "val"
        train_ratio: float = 0.9,
        max_samples: Optional[int] = None,
        normalize_pos: bool = True,
        inject_rotation: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing pickle files
            split: "train" or "val"
            train_ratio: Ratio of samples for training
            max_samples: Maximum samples to load (None = all)
            normalize_pos: Normalize positions to [-1, 1]
            inject_rotation: Add random rotation labels
            verbose: Print loading info
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.normalize_pos = normalize_pos
        self.inject_rotation = inject_rotation
        
        # Load all samples
        self.samples = self._load_all_pickles(data_dir, max_samples, verbose)
        
        # Train/val split
        n_total = len(self.samples)
        n_train = int(n_total * train_ratio)
        
        if split == "train":
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]
        
        if verbose:
            print(f"  {split}: {len(self.samples)} samples")
    
    def _load_all_pickles(self, data_dir: str, max_samples: Optional[int], verbose: bool) -> List:
        """Load all pickle files from directory."""
        pickle_files = sorted(glob.glob(os.path.join(data_dir, "*.pickle")))
        
        if verbose:
            print(f"\n=== Loading DiffPlace Dataset ===")
            print(f"  Directory: {data_dir}")
            print(f"  Found {len(pickle_files)} pickle files")
        
        all_samples = []
        for pf in pickle_files:
            with open(pf, 'rb') as f:
                samples = pickle.load(f)
                all_samples.extend(samples)
            
            if max_samples and len(all_samples) >= max_samples:
                all_samples = all_samples[:max_samples]
                break
        
        if verbose:
            print(f"  Total samples: {len(all_samples)}")
        
        return all_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a single sample with on-the-fly transformations.
        
        Returns PyG Data object with:
        - pos: (V, 2) ground truth positions
        - x: (V, 2) macro sizes (potentially rotated)
        - edge_index, edge_attr
        - is_ports: (V,) port mask
        - rot_label: (V,) rotation labels {0,1,2,3}
        - net_to_pin: (num_nets, max_pins) net structure
        - pin_to_macro: (P,) pin to macro mapping
        - pin_offsets: (P, 2) pin offsets from macro center
        """
        x, cond = self.samples[idx]
        
        V = x.shape[0]
        device = x.device
        
        # Clone to avoid modifying original
        pos = x.clone()
        sizes = cond.x.clone()
        edge_index = cond.edge_index.clone()
        edge_attr = cond.edge_attr.clone() if cond.edge_attr is not None else None
        is_ports = cond.is_ports.clone() if hasattr(cond, 'is_ports') else torch.zeros(V, dtype=torch.bool)
        
        # === 1. Position Normalization ===
        if self.normalize_pos:
            pos = self._normalize_positions(pos)
        
        # === 2. Rotation Injection ===
        if self.inject_rotation:
            rot_label = torch.randint(0, 4, (V,), dtype=torch.long)
            sizes = self._apply_rotation_to_sizes(sizes, rot_label)
        else:
            rot_label = torch.zeros(V, dtype=torch.long)
        
        # === 3. Edge-to-Net Conversion ===
        net_to_pin, pin_to_macro, pin_offsets = self._edge_to_net(edge_index, sizes, V)
        
        # Build output Data object
        data = Data(
            pos=pos,
            x=sizes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_ports=is_ports,
            rot_label=rot_label,
            net_to_pin=net_to_pin,
            pin_to_macro=pin_to_macro,
            pin_offsets=pin_offsets,
            num_nodes=V,
        )
        
        return data
    
    def _normalize_positions(self, pos: torch.Tensor) -> torch.Tensor:
        """Normalize positions to [-1, 1] range."""
        # Per-sample normalization
        min_vals = pos.min(dim=0).values
        max_vals = pos.max(dim=0).values
        range_vals = max_vals - min_vals
        range_vals = torch.clamp(range_vals, min=1e-6)
        
        # Normalize to [0, 1] then to [-1, 1]
        pos_norm = (pos - min_vals) / range_vals
        pos_norm = pos_norm * 2 - 1
        
        return pos_norm
    
    def _apply_rotation_to_sizes(
        self, sizes: torch.Tensor, rot_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotation to macro sizes.
        
        Rotation 0° or 180°: keep (w, h)
        Rotation 90° or 270°: swap to (h, w)
        """
        w, h = sizes[:, 0], sizes[:, 1]
        
        # Swap for 90° and 270° rotations
        should_swap = (rot_label == 1) | (rot_label == 3)
        
        new_w = torch.where(should_swap, h, w)
        new_h = torch.where(should_swap, w, h)
        
        return torch.stack([new_w, new_h], dim=1)
    
    def _edge_to_net(
        self, edge_index: torch.Tensor, sizes: torch.Tensor, num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert edge_index to net structure for Trinity guidance.
        
        Each edge becomes a 2-pin net.
        
        Returns:
            net_to_pin: (num_nets, max_pins=2) net to pin mapping
            pin_to_macro: (num_pins,) pin to macro mapping
            pin_offsets: (num_pins, 2) pin offsets from macro center
        """
        num_edges = edge_index.shape[1]
        
        # Each edge = 1 net with 2 pins
        num_nets = num_edges
        num_pins = num_edges * 2
        max_pins = 2
        
        # Net to pin: each net has pins [2*i, 2*i+1]
        net_to_pin = torch.arange(num_pins).view(num_nets, max_pins)
        
        # Pin to macro: pins map to source/target of edges
        src, dst = edge_index[0], edge_index[1]
        pin_to_macro = torch.zeros(num_pins, dtype=torch.long)
        pin_to_macro[0::2] = src  # Even pins = source
        pin_to_macro[1::2] = dst  # Odd pins = target
        
        # Pin offsets: random offsets within macro bounds
        half_sizes = sizes[pin_to_macro] / 2
        # Random offset in [-0.8*half, 0.8*half]
        pin_offsets = (torch.rand(num_pins, 2) * 2 - 1) * 0.8 * half_sizes
        
        return net_to_pin, pin_to_macro, pin_offsets


def collate_fn(data_list: List[Data]) -> Batch:
    """
    Custom collate function with follow_batch for proper index handling.
    """
    return Batch.from_data_list(
        data_list,
        follow_batch=['net_to_pin'],
    )


def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """Create DataLoader for DiffPlace synthetic data."""
    dataset = DiffPlacePickleDataset(data_dir, split=split, **dataset_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_sample(data: Data, ax=None, show_rotation: bool = True):
    """
    Visualize a single placement sample.
    
    Args:
        data: PyG Data object with pos, x (sizes), rot_label
        ax: Matplotlib axis (creates new if None)
        show_rotation: Color macros by rotation
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    pos = data.pos.numpy() if hasattr(data.pos, 'numpy') else data.pos
    sizes = data.x.numpy() if hasattr(data.x, 'numpy') else data.x
    
    # Rotation colors
    rot_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # 0°, 90°, 180°, 270°
    
    if hasattr(data, 'rot_label'):
        rot_label = data.rot_label.numpy() if hasattr(data.rot_label, 'numpy') else data.rot_label
    else:
        rot_label = np.zeros(len(pos))
    
    for i, (p, s, r) in enumerate(zip(pos, sizes, rot_label)):
        x, y = p
        w, h = s
        
        # Rectangle centered at (x, y)
        rect = patches.Rectangle(
            (x - w/2, y - h/2), w, h,
            linewidth=1,
            edgecolor='black',
            facecolor=rot_colors[int(r)] if show_rotation else '#3498db',
            alpha=0.7,
        )
        ax.add_patch(rect)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=-1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'N={len(pos)} macros')
    
    return ax


def debug_batch(batch: Batch, output_path: str = "debug_batch.png"):
    """
    Visualize a batch of samples for debugging.
    
    Args:
        batch: Batched Data object
        output_path: Path to save visualization
    """
    # Unbatch
    data_list = batch.to_data_list()
    n_samples = min(len(data_list), 4)
    
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    for i, (data, ax) in enumerate(zip(data_list[:n_samples], axes)):
        visualize_sample(data, ax)
        ax.set_title(f'Sample {i}: N={data.num_nodes}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved debug visualization to: {output_path}")
