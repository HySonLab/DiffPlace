"""
DiffPlace Data Transformation Module

Handles preprocessing of netlist data for VectorGNN backbone:
- Separates position (x, y) from node features (width, height)
- Normalizes vectors to [-1, 1]
- Ensures edge_index is properly formatted (bidirectional)
- Adds .pos attribute for relative position computation

Data Format:
============
Input (from load_graph_data):
    x: (V, 2) - placement positions (already normalized to [-1, 1])
    cond: PyG Data object with:
        .x: (V, 2) - node sizes (width, height), normalized
        .edge_index: (2, E) - netlist connections (bidirectional)
        .edge_attr: (E, 4) - edge attributes
        .is_ports: (V,) - boolean mask for ports

Output (for DiffPlaceModel):
    x: (B, V, 2) - noisy positions for diffusion
    cond: PyG Data object with:
        .x: (V, node_features) - node features (sizes)
        .pos: (V, 2) - clean positions (for reference/ports)
        .node_sizes: (V, 2) - explicit sizes for guidance
        .edge_index: (2, E) - edges
        .edge_attr: (E, edge_features) - edge attributes
        .is_ports: (V,) - port mask
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Optional, Tuple, List, Union


class DiffPlaceTransform:
    """
    Transform raw netlist data for DiffPlace model.
    
    Key operations:
    1. Separate positions from features
    2. Normalize positions to [-1, 1]
    3. Ensure bidirectional edges
    4. Add positional attributes for VectorGNN
    """
    
    def __init__(
        self,
        pos_scale: float = 1.0,
        normalize_positions: bool = True,
        ensure_bidirectional: bool = True,
        add_size_features: bool = True,
    ):
        """
        Args:
            pos_scale: Scale factor for positions after normalization
            normalize_positions: Whether to normalize positions to [-1, 1]
            ensure_bidirectional: Whether to ensure edges are bidirectional
            add_size_features: Whether to add node sizes as separate attribute
        """
        self.pos_scale = pos_scale
        self.normalize_positions = normalize_positions
        self.ensure_bidirectional = ensure_bidirectional
        self.add_size_features = add_size_features
    
    def __call__(
        self, 
        x: torch.Tensor, 
        cond: Data
    ) -> Tuple[torch.Tensor, Data]:
        """
        Transform placement data.
        
        Args:
            x: (V, 2) or (B, V, 2) position tensor
            cond: PyG Data object with netlist info
        
        Returns:
            x: transformed positions
            cond: transformed conditioning data
        """
        # Handle batched input
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, V, 2)
        
        B, V, D = x.shape
        assert D == 2, f"Position dimension must be 2, got {D}"
        
        # Clone cond to avoid modifying original
        cond = self._clone_cond(cond)
        
        # Store original positions for ports
        cond.pos = x[0].clone()  # (V, 2) - reference positions
        
        # Normalize positions if needed
        if self.normalize_positions:
            x = self._normalize_positions(x)
        
        # Scale positions
        x = x * self.pos_scale
        
        # Store node sizes explicitly
        if self.add_size_features:
            cond.node_sizes = cond.x.clone()  # (V, 2) - widths and heights
        
        # Ensure bidirectional edges
        if self.ensure_bidirectional:
            cond = self._ensure_bidirectional_edges(cond)
        
        return x, cond
    
    def _clone_cond(self, cond: Data) -> Data:
        """Create a copy of conditioning data."""
        new_cond = Data()
        # Handle both PyG Data objects and mock objects
        if hasattr(cond, 'keys') and callable(cond.keys):
            keys = list(cond.keys())
        else:
            keys = [k for k in dir(cond) if not k.startswith('_') and not callable(getattr(cond, k))]
        
        for key in keys:
            value = getattr(cond, key)
            if torch.is_tensor(value):
                setattr(new_cond, key, value.clone())
            else:
                setattr(new_cond, key, value)
        return new_cond
    
    def _normalize_positions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize positions to [-1, 1] range.
        
        Args:
            x: (B, V, 2) positions
        
        Returns:
            normalized: (B, V, 2) positions in [-1, 1]
        """
        # Check if already normalized
        if x.min() >= -1.5 and x.max() <= 1.5:
            return x  # Already approximately normalized
        
        # Normalize per batch
        x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        
        # Avoid division by zero
        scale = x_max - x_min
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        
        # Normalize to [0, 1] then to [-1, 1]
        x_normalized = (x - x_min) / scale
        x_normalized = 2 * x_normalized - 1
        
        return x_normalized
    
    def _ensure_bidirectional_edges(self, cond: Data) -> Data:
        """
        Ensure edges are bidirectional (required for some operations).
        
        Args:
            cond: Data object with edge_index
        
        Returns:
            cond: Data object with bidirectional edges
        """
        edge_index = cond.edge_index
        E = edge_index.shape[1]
        
        # Check if already bidirectional
        if E % 2 == 0:
            # Simple check: assume first half are forward, second half are reverse
            src, dst = edge_index[0], edge_index[1]
            if torch.equal(src[:E//2], dst[E//2:]) and torch.equal(dst[:E//2], src[E//2:]):
                return cond  # Already bidirectional
        
        # Make bidirectional
        src, dst = edge_index[0], edge_index[1]
        
        # Create reverse edges
        reverse_edge_index = torch.stack([dst, src], dim=0)
        
        # Concatenate (avoid duplicates by using unique)
        all_edges = torch.cat([edge_index, reverse_edge_index], dim=1)
        
        # Remove duplicates
        all_edges_sorted, _ = torch.sort(all_edges, dim=0)
        unique_mask = torch.ones(all_edges.shape[1], dtype=torch.bool)
        for i in range(1, all_edges.shape[1]):
            if torch.equal(all_edges_sorted[:, i], all_edges_sorted[:, i-1]):
                unique_mask[i] = False
        
        # Actually, let's just keep all edges (PyG handles duplicates)
        cond.edge_index = all_edges
        
        # Duplicate edge attributes if they exist
        if hasattr(cond, 'edge_attr') and cond.edge_attr is not None:
            cond.edge_attr = torch.cat([cond.edge_attr, cond.edge_attr], dim=0)
        
        return cond


class DiffPlaceCollate:
    """
    Custom collate function for DiffPlace dataloader.
    
    Handles:
    - Batching position tensors (B, V, 2)
    - Creating proper PyG batch objects
    - Preserving mask information
    """
    
    def __init__(
        self,
        transform: Optional[DiffPlaceTransform] = None,
        device: str = "cpu",
    ):
        """
        Args:
            transform: Optional transform to apply
            device: Device to place tensors on
        """
        self.transform = transform or DiffPlaceTransform()
        self.device = device
    
    def __call__(
        self, 
        batch: List[Tuple[torch.Tensor, Data]]
    ) -> Tuple[torch.Tensor, Data]:
        """
        Collate a batch of (x, cond) pairs.
        
        Args:
            batch: List of (x, cond) tuples where x is (V, 2)
        
        Returns:
            x_batch: (B, V, 2) batched positions
            cond: Single PyG Data object (shared across batch)
        """
        x_list = [item[0] for item in batch]
        cond_list = [item[1] for item in batch]
        
        # For graph data, we assume all samples share the same graph structure
        # (common in chip placement where netlist is fixed)
        # So we just use the first cond
        cond = cond_list[0]
        
        # Stack positions
        x_batch = torch.stack(x_list, dim=0)  # (B, V, 2)
        
        # Apply transform
        x_batch, cond = self.transform(x_batch, cond)
        
        # Move to device
        x_batch = x_batch.to(self.device)
        cond = cond.to(self.device)
        
        return x_batch, cond


class DiffPlaceGraphDataLoader:
    """
    DataLoader for DiffPlace that properly handles graph data.
    
    Key features:
    - Separates positions from node features
    - Applies DiffPlaceTransform
    - Compatible with DiffPlaceModel
    """
    
    def __init__(
        self,
        train_dataset: List[Tuple[torch.Tensor, Data]],
        val_dataset: List[Tuple[torch.Tensor, Data]],
        train_batch_size: int = 16,
        val_batch_size: int = 1,
        train_device: str = "cuda",
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        transform: Optional[DiffPlaceTransform] = None,
    ):
        """
        Args:
            train_dataset: List of (x, cond) tuples for training
            val_dataset: List of (x, cond) tuples for validation
            train_batch_size: Training batch size
            val_batch_size: Validation batch size
            train_device: Device for training
            shuffle_train: Whether to shuffle training data
            shuffle_val: Whether to shuffle validation data
            transform: Optional transform to apply
        """
        self.train_set = train_dataset
        self.val_set = val_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.device = train_device
        self.shuffle = {"train": shuffle_train, "val": shuffle_val}
        self.transform = transform or DiffPlaceTransform()
        
        self.current_idx = {"train": 0, "val": 0}
        self._display_cache = {"train": None, "val": None}
    
    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, Data]:
        """
        Get a batch of data.
        
        Args:
            split: "train" or "val"
        
        Returns:
            x: (B, V, 2) positions
            cond: PyG Data object
        """
        assert split in ("train", "val")
        
        dataset = self.train_set if split == "train" else self.val_set
        batch_size = self.train_batch_size if split == "train" else self.val_batch_size
        
        if self.shuffle[split]:
            idx = torch.randint(0, len(dataset), (1,)).item()
        else:
            idx = self.current_idx[split]
            self.current_idx[split] = (self.current_idx[split] + 1) % len(dataset)
        
        x, cond = dataset[idx]
        
        # Ensure x is float tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        
        # Expand for batch
        x = x.view(1, *x.shape).expand(batch_size, -1, -1).clone()
        
        # Apply transform
        x, cond = self.transform(x, cond)
        
        # Move to device
        x = x.to(self.device)
        cond = cond.to(self.device)
        
        return x, cond
    
    def get_display_batch(
        self, 
        num_samples: int, 
        split: str = "val"
    ) -> Tuple[torch.Tensor, Data]:
        """Get a fixed batch for visualization."""
        if self._display_cache[split] is None:
            self._display_cache[split] = self.get_batch(split)
        
        x, cond = self._display_cache[split]
        return x[:num_samples], cond
    
    def reset_idx(self, split: str):
        """Reset iterator index for non-shuffle mode."""
        self.current_idx[split] = 0
    
    def get_train_size(self) -> int:
        return len(self.train_set)
    
    def get_val_size(self) -> int:
        return len(self.val_set)


def prepare_cond_for_diffplace(cond: Data) -> Data:
    """
    Prepare a conditioning Data object for DiffPlaceModel.
    
    Ensures all required attributes are present:
    - .x: (V, 2) node features (sizes)
    - .edge_index: (2, E) bidirectional edges
    - .edge_attr: (E, F) edge attributes
    - .is_ports: (V,) boolean mask
    - .pos: (V, 2) reference positions (optional)
    - .node_sizes: (V, 2) explicit sizes (optional)
    
    Args:
        cond: Original PyG Data object
    
    Returns:
        cond: Prepared Data object
    """
    # Validate required attributes
    assert hasattr(cond, 'x'), "cond must have 'x' attribute (node features)"
    assert hasattr(cond, 'edge_index'), "cond must have 'edge_index' attribute"
    
    # Add default is_ports if missing
    if not hasattr(cond, 'is_ports') or cond.is_ports is None:
        V = cond.x.shape[0]
        cond.is_ports = torch.zeros(V, dtype=torch.bool, device=cond.x.device)
    
    # Add edge_attr if missing
    if not hasattr(cond, 'edge_attr') or cond.edge_attr is None:
        E = cond.edge_index.shape[1]
        cond.edge_attr = torch.zeros(E, 4, dtype=torch.float32, device=cond.edge_index.device)
    
    # Add node_sizes if not present (copy from x)
    if not hasattr(cond, 'node_sizes'):
        cond.node_sizes = cond.x.clone()
    
    return cond


def normalize_placement(
    x: torch.Tensor, 
    chip_size: Optional[Tuple[float, float]] = None,
    chip_offset: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Normalize placement coordinates to [-1, 1] range.
    
    Args:
        x: (..., 2) positions in absolute coordinates
        chip_size: (width, height) of chip canvas
        chip_offset: (x_offset, y_offset) of chip origin
    
    Returns:
        x_normalized: (..., 2) positions in [-1, 1]
    """
    if chip_size is not None:
        chip_size = torch.tensor(chip_size, dtype=x.dtype, device=x.device)
        if chip_offset is not None:
            chip_offset = torch.tensor(chip_offset, dtype=x.dtype, device=x.device)
            x = x - chip_offset
        x = 2 * (x / chip_size) - 1
    else:
        # Auto-normalize based on data range
        x_min = x.min(dim=-2, keepdim=True)[0]
        x_max = x.max(dim=-2, keepdim=True)[0]
        scale = x_max - x_min
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        x = 2 * ((x - x_min) / scale) - 1
    
    return x


def denormalize_placement(
    x: torch.Tensor,
    chip_size: Tuple[float, float],
    chip_offset: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Convert normalized positions back to absolute coordinates.
    
    Args:
        x: (..., 2) positions in [-1, 1]
        chip_size: (width, height) of chip canvas
        chip_offset: (x_offset, y_offset) of chip origin
    
    Returns:
        x_abs: (..., 2) positions in absolute coordinates
    """
    chip_size = torch.tensor(chip_size, dtype=x.dtype, device=x.device)
    x = chip_size * (x + 1) / 2
    
    if chip_offset is not None:
        chip_offset = torch.tensor(chip_offset, dtype=x.dtype, device=x.device)
        x = x + chip_offset
    
    return x


# Convenience function to create transform with common settings
def create_diffplace_transform(
    normalize: bool = True,
    ensure_bidirectional: bool = True,
) -> DiffPlaceTransform:
    """Create a DiffPlaceTransform with common settings."""
    return DiffPlaceTransform(
        pos_scale=1.0,
        normalize_positions=normalize,
        ensure_bidirectional=ensure_bidirectional,
        add_size_features=True,
    )


# Convenience function to wrap existing GraphDataLoader
def wrap_graph_dataloader(
    original_loader,
    transform: Optional[DiffPlaceTransform] = None,
):
    """
    Wrap an existing GraphDataLoader to apply DiffPlaceTransform.
    
    This allows using existing data loading code with DiffPlace.
    """
    transform = transform or create_diffplace_transform()
    
    class WrappedLoader:
        def __init__(self, loader, transform):
            self._loader = loader
            self._transform = transform
        
        def get_batch(self, split="train"):
            x, cond = self._loader.get_batch(split)
            cond = prepare_cond_for_diffplace(cond)
            x, cond = self._transform(x, cond)
            return x, cond
        
        def get_display_batch(self, num_samples, split="val"):
            x, cond = self._loader.get_display_batch(num_samples, split)
            cond = prepare_cond_for_diffplace(cond)
            x, cond = self._transform(x, cond)
            return x, cond
        
        def reset_idx(self, split):
            if hasattr(self._loader, 'reset_idx'):
                self._loader.reset_idx(split)
        
        def get_train_size(self):
            return self._loader.get_train_size()
        
        def get_val_size(self):
            return self._loader.get_val_size()
        
        def __getattr__(self, name):
            return getattr(self._loader, name)
    
    return WrappedLoader(original_loader, transform)

