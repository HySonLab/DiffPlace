"""
ISPD2005 Bookshelf Dataset for DiffPlace

Parses standard Bookshelf format (.nodes, .nets, .pl, .scl)
and converts to PyG Data for training.
"""

import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BookshelfParser:
    """Parser for ISPD2005 Bookshelf format."""
    
    def __init__(self, benchmark_dir: str, benchmark_name: str):
        """
        Args:
            benchmark_dir: Directory containing benchmark files
            benchmark_name: Name of benchmark (e.g., 'adaptec1')
        """
        self.benchmark_dir = benchmark_dir
        self.benchmark_name = benchmark_name
        self.base_path = os.path.join(benchmark_dir, benchmark_name, benchmark_name)
        
        # Data containers
        self.nodes: Dict[str, Dict] = {}  # name -> {w, h, is_terminal}
        self.node_names: List[str] = []   # Ordered list
        self.name_to_idx: Dict[str, int] = {}
        
        self.nets: List[Dict] = []  # [{name, pins: [(node_name, offset_x, offset_y), ...]}]
        self.placements: Dict[str, Tuple] = {}  # name -> (x, y, orientation)
        
        self.die_area = None  # (width, height)
    
    def parse(self) -> None:
        """Parse all Bookshelf files."""
        self._parse_nodes()
        self._parse_nets()
        self._parse_placement()
        self._parse_scl()
    
    def _parse_nodes(self) -> None:
        """Parse .nodes file for node dimensions."""
        nodes_file = f"{self.base_path}.nodes"
        
        with open(nodes_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('UCLA'):
                    continue
                if line.startswith('NumNodes') or line.startswith('NumTerminals'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    width = float(parts[1])
                    height = float(parts[2])
                    is_terminal = len(parts) > 3 and 'terminal' in parts[3].lower()
                    
                    self.nodes[name] = {
                        'width': width,
                        'height': height,
                        'is_terminal': is_terminal,
                    }
                    self.name_to_idx[name] = len(self.node_names)
                    self.node_names.append(name)
    
    def _parse_nets(self) -> None:
        """Parse .nets file for connectivity."""
        nets_file = f"{self.base_path}.nets"
        
        current_net = None
        
        with open(nets_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('UCLA'):
                    continue
                if line.startswith('NumNets') or line.startswith('NumPins'):
                    continue
                
                if line.startswith('NetDegree'):
                    # New net
                    parts = line.split()
                    net_name = parts[2] if len(parts) > 2 else f"net_{len(self.nets)}"
                    current_net = {'name': net_name, 'pins': []}
                    self.nets.append(current_net)
                elif current_net is not None:
                    # Pin entry
                    parts = line.split()
                    if len(parts) >= 4:
                        node_name = parts[0]
                        # Skip I/O indicator (parts[1])
                        # Offset is typically parts[2:4] with ':' separator
                        try:
                            offset_x = float(parts[3].replace(':', ''))
                            offset_y = float(parts[4]) if len(parts) > 4 else 0.0
                        except (ValueError, IndexError):
                            offset_x, offset_y = 0.0, 0.0
                        
                        if node_name in self.name_to_idx:
                            current_net['pins'].append((node_name, offset_x, offset_y))
    
    def _parse_placement(self) -> None:
        """Parse .pl file for initial placement."""
        # Try lg.pl first (legalized), then eplace-ip.pl, then .pl
        pl_files = [
            f"{self.base_path}.lg.pl",
            f"{self.base_path}.eplace-ip.pl", 
            f"{self.base_path}.pl",
        ]
        
        pl_file = None
        for pf in pl_files:
            if os.path.exists(pf):
                pl_file = pf
                break
        
        if pl_file is None:
            raise FileNotFoundError(f"No placement file found for {self.benchmark_name}")
        
        print(f"  Using placement: {os.path.basename(pl_file)}")
        
        with open(pl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('UCLA'):
                    continue
                
                # Format: name x y : orientation
                parts = line.split()
                if len(parts) >= 4 and parts[2] == ':':
                    name = parts[0]
                    x = float(parts[1])
                    y = float(parts[3])  # Skip ':'
                    orientation = parts[4] if len(parts) > 4 else 'N'
                    self.placements[name] = (x, y, orientation)
                elif len(parts) >= 3:
                    name = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    orientation = parts[4] if len(parts) > 4 else 'N'
                    self.placements[name] = (x, y, orientation)
    
    def _parse_scl(self) -> None:
        """Parse .scl file for die area (optional)."""
        scl_file = f"{self.base_path}.scl"
        if not os.path.exists(scl_file):
            return
        
        max_x, max_y = 0, 0
        with open(scl_file, 'r') as f:
            for line in f:
                if 'Coordinate' in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        max_y = max(max_y, int(match.group(1)))
                if 'Subroworigin' in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        max_x = max(max_x, int(match.group(1)))
        
        if max_x > 0 and max_y > 0:
            self.die_area = (max_x * 2, max_y * 2)  # Rough estimate
    
    def to_pyg_data(self, max_nodes: Optional[int] = None) -> Data:
        """
        Convert parsed data to PyG Data object.
        
        Args:
            max_nodes: Limit number of nodes (for memory)
        
        Returns:
            PyG Data compatible with DiffPlace
        """
        # Filter to nodes with placement
        valid_nodes = [n for n in self.node_names if n in self.placements]
        
        if max_nodes and len(valid_nodes) > max_nodes:
            # Sample subset
            np.random.seed(42)
            indices = np.random.choice(len(valid_nodes), max_nodes, replace=False)
            valid_nodes = [valid_nodes[i] for i in sorted(indices)]
        
        # Remap indices
        new_idx = {name: i for i, name in enumerate(valid_nodes)}
        V = len(valid_nodes)
        
        if V == 0:
            raise ValueError("No valid nodes with placement found")
        
        # Build tensors
        positions = torch.zeros(V, 2)
        sizes = torch.zeros(V, 2)
        is_terminal = torch.zeros(V, dtype=torch.bool)
        rot_label = torch.zeros(V, dtype=torch.long)  # Will map orientation
        
        # Orientation mapping
        orient_map = {'N': 0, 'S': 2, 'E': 1, 'W': 3, 'FN': 0, 'FS': 2, 'FE': 1, 'FW': 3}
        
        for i, name in enumerate(valid_nodes):
            node = self.nodes[name]
            x, y, orient = self.placements[name]
            
            positions[i] = torch.tensor([x, y])
            sizes[i] = torch.tensor([node['width'], node['height']])
            is_terminal[i] = node.get('is_terminal', False)
            rot_label[i] = orient_map.get(orient, 0)
        
        # Normalize positions to [-1, 1]
        pos_min = positions.min(dim=0).values
        pos_max = positions.max(dim=0).values
        pos_range = pos_max - pos_min
        pos_range = torch.clamp(pos_range, min=1.0)
        positions = (positions - pos_min) / pos_range * 2 - 1
        
        # Normalize sizes relative to die
        sizes = sizes / pos_range.unsqueeze(0).expand(V, 2) * 0.1
        
        # Build edge_index from nets
        edges_src = []
        edges_dst = []
        
        for net in self.nets:
            pins = [(new_idx[p[0]], p[1], p[2]) 
                    for p in net['pins'] if p[0] in new_idx]
            
            if len(pins) >= 2:
                # Star topology: first pin connects to all others
                for i in range(1, len(pins)):
                    edges_src.extend([pins[0][0], pins[i][0]])
                    edges_dst.extend([pins[i][0], pins[0][0]])
        
        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            # Edge attributes (dummy for now)
            E = edge_index.shape[1]
            edge_attr = torch.randn(E, 4) * 0.1
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 4)
        
        # Build net structure for Trinity guidance
        net_to_pin, pin_to_macro, pin_offsets = self._build_net_structure(
            valid_nodes, new_idx
        )
        
        return Data(
            pos=positions,
            x=sizes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_ports=is_terminal,
            rot_label=rot_label,
            net_to_pin=net_to_pin,
            pin_to_macro=pin_to_macro,
            pin_offsets=pin_offsets,
            num_nodes=V,
            benchmark=self.benchmark_name,
        )
    
    def _build_net_structure(
        self, valid_nodes: List[str], new_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build net structure for Trinity guidance."""
        all_pins = []
        net_pins = []
        max_pins = 0
        
        for net in self.nets:
            pins = [(new_idx[p[0]], p[1], p[2]) 
                    for p in net['pins'] if p[0] in new_idx]
            
            if len(pins) >= 2:
                start_idx = len(all_pins)
                for macro_idx, off_x, off_y in pins:
                    all_pins.append((macro_idx, off_x, off_y))
                
                net_pins.append(list(range(start_idx, start_idx + len(pins))))
                max_pins = max(max_pins, len(pins))
        
        if not net_pins:
            # Dummy structure
            return (
                torch.zeros(1, 2, dtype=torch.long),
                torch.zeros(2, dtype=torch.long),
                torch.zeros(2, 2),
            )
        
        # Pad net_to_pin
        num_nets = len(net_pins)
        net_to_pin = torch.full((num_nets, max_pins), -1, dtype=torch.long)
        for i, pins in enumerate(net_pins):
            net_to_pin[i, :len(pins)] = torch.tensor(pins)
        
        # Pin to macro and offsets
        num_pins = len(all_pins)
        pin_to_macro = torch.zeros(num_pins, dtype=torch.long)
        pin_offsets = torch.zeros(num_pins, 2)
        
        for i, (macro_idx, off_x, off_y) in enumerate(all_pins):
            pin_to_macro[i] = macro_idx
            pin_offsets[i] = torch.tensor([off_x, off_y]) * 0.001  # Scale down
        
        return net_to_pin, pin_to_macro, pin_offsets


class ISPDDataset(Dataset):
    """
    Dataset for ISPD2005 benchmarks.
    
    Supports multiple benchmarks and subsampling for memory.
    """
    
    def __init__(
        self,
        benchmark_dir: str,
        benchmarks: Optional[List[str]] = None,
        max_nodes: int = 50000,
        augment: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            benchmark_dir: Base directory (e.g., data/ispd2005)
            benchmarks: List of benchmark names (default: all)
            max_nodes: Max nodes per sample (subsample if larger)
            augment: Apply random augmentation
            verbose: Print loading info
        """
        super().__init__()
        self.benchmark_dir = benchmark_dir
        self.max_nodes = max_nodes
        self.augment = augment
        
        # Default benchmarks
        if benchmarks is None:
            benchmarks = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4',
                         'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4']
        
        # Only use existing benchmarks
        benchmarks = [b for b in benchmarks 
                     if os.path.isdir(os.path.join(benchmark_dir, b))]
        
        if verbose:
            print(f"\n=== Loading ISPD2005 Dataset ===")
            print(f"  Directory: {benchmark_dir}")
            print(f"  Benchmarks: {benchmarks}")
        
        # Parse all benchmarks
        self.data_list = []
        for name in benchmarks:
            try:
                if verbose:
                    print(f"  Parsing {name}...", end=' ')
                
                parser = BookshelfParser(benchmark_dir, name)
                parser.parse()
                data = parser.to_pyg_data(max_nodes=max_nodes)
                self.data_list.append(data)
                
                if verbose:
                    print(f"V={data.num_nodes}, E={data.edge_index.shape[1]}")
                    
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
        
        if verbose:
            print(f"  Total samples: {len(self.data_list)}")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Data:
        data = self.data_list[idx].clone()
        
        if self.augment:
            # Random rotation (0, 90, 180, 270)
            if torch.rand(1).item() < 0.5:
                data.pos = data.pos.flip(1)  # Swap x, y
                data.x = data.x.flip(1)      # Swap w, h
                data.rot_label = (data.rot_label + 1) % 4
            
            # Random flip
            if torch.rand(1).item() < 0.5:
                data.pos[:, 0] = -data.pos[:, 0]
            
            if torch.rand(1).item() < 0.5:
                data.pos[:, 1] = -data.pos[:, 1]
        
        return data


def collate_ispd(data_list: List[Data]) -> Data:
    """Simple collate - returns first item for batch_size=1."""
    if len(data_list) == 1:
        return data_list[0]
    else:
        # For batch > 1, use PyG Batch
        from torch_geometric.data import Batch
        return Batch.from_data_list(data_list)


def create_ispd_dataloader(
    benchmark_dir: str,
    benchmarks: Optional[List[str]] = None,
    batch_size: int = 1,
    max_nodes: int = 50000,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for ISPD dataset."""
    dataset = ISPDDataset(
        benchmark_dir=benchmark_dir,
        benchmarks=benchmarks,
        max_nodes=max_nodes,
        **kwargs,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ispd,
    )


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    
    benchmark_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ispd2005"  # Path to ISPD benchmarks
    
    print("=" * 60)
    print("ISPD2005 Dataset Test")
    print("=" * 60)
    
    # Test single benchmark
    dataset = ISPDDataset(
        benchmark_dir=benchmark_dir,
        benchmarks=['adaptec1'],
        max_nodes=10000,
        verbose=True,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n=== Sample ===")
        print(f"  pos: {sample.pos.shape}")
        print(f"  x (sizes): {sample.x.shape}")
        print(f"  edge_index: {sample.edge_index.shape}")
        print(f"  rot_label: {sample.rot_label.shape}")
        print(f"  is_ports: {sample.is_ports.sum().item()} terminals")
        print(f"  net_to_pin: {sample.net_to_pin.shape}")
        
        print("\n✓ ISPD Dataset ready!")
    else:
        print("\n✗ No data loaded!")
