# Synthetic Training Data Format

This directory should contain `.pickle` files for training DiffPlace.

## Data Format

Each `.pickle` file contains a **list of tuples**:

```python
[
    (positions, graph_data),
    (positions, graph_data),
    ...
]
```

### Tuple Structure

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `positions` | `torch.Tensor` | `(V, 2)` | Ground truth macro center coordinates (x, y) |
| `graph_data` | `torch_geometric.data.Data` | - | Graph structure with node/edge features |

### Graph Data Attributes (Required)

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `x` | `torch.Tensor` | `(V, 2)` | Macro sizes (width, height) |
| `edge_index` | `torch.LongTensor` | `(2, E)` | Graph connectivity |
| `is_ports` | `torch.BoolTensor` | `(V,)` | Fixed node mask (True = fixed/port) |

### Graph Data Attributes (Optional)

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `edge_attr` | `torch.Tensor` | `(E, 4)` | Pin offsets: (src_dx, src_dy, dst_dx, dst_dy) |
| `node_sizes` | `torch.Tensor` | `(V, 2)` | Alternative to `x` for sizes |

## Example

```python
import torch
import pickle
from torch_geometric.data import Data

# Create sample data
V = 50  # Number of macros
E = 100  # Number of edges

positions = torch.rand(V, 2) * 1000  # Positions in [0, 1000]
graph = Data(
    x=torch.rand(V, 2) * 50 + 10,  # Sizes: width, height in [10, 60]
    edge_index=torch.randint(0, V, (2, E)),  # Random connectivity
    is_ports=torch.zeros(V, dtype=torch.bool),  # No fixed nodes
)

# Save to pickle
samples = [(positions, graph) for _ in range(100)]
with open('data/synthetic_data/batch_001.pickle', 'wb') as f:
    pickle.dump(samples, f)
```

## Directory Structure

```
data/synthetic_data/
├── batch_001.pickle  # 100 samples
├── batch_002.pickle  # 100 samples
├── ...
└── README.md
```

## Coordinate System

- **Positions**: Macro center coordinates
- **Sizes**: Full width and height (not half-sizes)
- **Normalization**: Training will normalize positions to [-1, 1] automatically

## Notes

- Each pickle file typically contains ~100 samples for efficient loading
- Files are split 90/10 for train/val automatically
- Rotation augmentation is applied on-the-fly during training
