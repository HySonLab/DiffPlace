import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OverlapLoss(nn.Module):
    """
    Differentiable pairwise macro overlap loss (fully vectorized).

    Assumes positions are **macro centers** and sizes are (w, h) in the same units.
    Computes sum of pairwise overlap areas across all (i < j).
    """

    def __init__(
        self,
        positive_fn: str = "relu",
        softplus_beta: float = 10.0,
    ):
        super().__init__()
        if positive_fn not in ("relu", "softplus"):
            raise ValueError(f"positive_fn must be 'relu' or 'softplus', got: {positive_fn}")
        self.positive_fn = positive_fn
        self.softplus_beta = float(softplus_beta)

    def _pos(self, x: torch.Tensor) -> torch.Tensor:
        if self.positive_fn == "relu":
            return F.relu(x)
        # softplus
        return F.softplus(x, beta=self.softplus_beta)

    def forward(
        self,
        coords: torch.Tensor,
        sizes: torch.Tensor,
        macro_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) or (B, N, 2) macro centers
            sizes:  (N, 2) or (B, N, 2) macro sizes (w, h)
            macro_mask: optional (N,) bool mask to select a subset (e.g., movable macros)

        Returns:
            Scalar tensor: sum of pairwise overlap areas over all (i < j) (and over batch, if any).
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # (1, N, 2)
        if sizes.dim() == 2:
            sizes = sizes.unsqueeze(0)  # (1, N, 2)
        # IMPORTANT: sizes are not variables in placement; avoid gradients on (w, h).
        sizes = sizes.detach()
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError(f"coords must be (N,2) or (B,N,2), got {tuple(coords.shape)}")
        if sizes.dim() != 3 or sizes.size(-1) != 2:
            raise ValueError(f"sizes must be (N,2) or (B,N,2), got {tuple(sizes.shape)}")
        if coords.shape[:2] != sizes.shape[:2]:
            raise ValueError(f"coords and sizes must match on (B,N); got {tuple(coords.shape)} vs {tuple(sizes.shape)}")

        B, N, _ = coords.shape
        device = coords.device

        if macro_mask is not None:
            if macro_mask.dim() != 1 or macro_mask.numel() != N:
                raise ValueError(f"macro_mask must be (N,), got {tuple(macro_mask.shape)} for N={N}")
            idx = torch.where(macro_mask.to(device=device))[0]
            # If 0/1 macro, overlap is zero.
            if idx.numel() < 2:
                return coords.new_zeros(())
            coords = coords.index_select(1, idx)
            sizes = sizes.index_select(1, idx)
            N = coords.shape[1]

        x = coords[..., 0]
        y = coords[..., 1]
        w = sizes[..., 0]
        h = sizes[..., 1]

        x1 = x - 0.5 * w
        x2 = x + 0.5 * w
        y1 = y - 0.5 * h
        y2 = y + 0.5 * h

        # Pairwise intersections (B, N, N)
        left = torch.maximum(x1.unsqueeze(-1), x1.unsqueeze(-2))
        right = torch.minimum(x2.unsqueeze(-1), x2.unsqueeze(-2))
        bottom = torch.maximum(y1.unsqueeze(-1), y1.unsqueeze(-2))
        top = torch.minimum(y2.unsqueeze(-1), y2.unsqueeze(-2))

        dx = self._pos(right - left)
        dy = self._pos(top - bottom)
        inter = dx * dy  # (B, N, N)

        # Sum only i < j to avoid double-counting and ignore diagonal.
        triu = torch.triu(torch.ones((N, N), device=device, dtype=inter.dtype), diagonal=1)
        loss = (inter * triu).sum()
        return loss

