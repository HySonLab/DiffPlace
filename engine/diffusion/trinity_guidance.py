

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# ============================================================================
# FORCE A: ELECTROSTATIC DENSITY POTENTIAL
# ============================================================================

class ElectrostaticDensity(nn.Module):
    r"""
    Electrostatic-based density spreading force.
    
    Theory (DREAMPlace-style):
    -------------------------
    Treat macros as charged particles that repel each other.
    
    .. math::
        \rho(x, y) = \sum_i q_i \cdot G_\sigma(x - x_i, y - y_i)
        
    where :math:`G_\sigma` is a Gaussian kernel and :math:`q_i` is the area of macro i.
    
    The potential energy:
    
    .. math::
        E_{density} = \int \int (\rho(x,y) - \rho_{target})^2 \, dx \, dy
        
    Gradient pushes macros away from high-density regions.
    
    Complexity: O(N + M² log M²)
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        sigma: float = 2.0,
        target_density: float = 1.0,
        boundary_weight: float = 10.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        self.target_density = target_density
        self.boundary_weight = boundary_weight
        
        # Gaussian kernel for smoothing
        self.register_buffer('gaussian_kernel', self._make_gaussian_kernel(sigma))
        
    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        ksize = int(6 * sigma) | 1
        ksize = max(ksize, 3)
        x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: (B, N, 2) in [-1, 1]
            sizes: (B, N, 2) or (N, 2)
            mask: (B, N) bool - True = skip
            
        Returns:
            potential: (B,) scalar
            density_map: (B, 1, M, M) for visualization
        """
        B, N, _ = positions.shape
        M = self.grid_size
        device = positions.device
        
        if sizes.dim() == 2:
            sizes = sizes.unsqueeze(0).expand(B, -1, -1)
            
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(B, -1)
        keep = ~mask if mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
        
        # Scatter density
        density = self._scatter_density(positions, sizes, keep)  # (B, 1, M, M)
        
        # Gaussian smooth
        kernel = self.gaussian_kernel.to(device, density.dtype)
        padding = kernel.shape[-1] // 2
        density_smooth = F.conv2d(density, kernel, padding=padding)
        
        # Overflow penalty
        overflow = F.relu(density_smooth - self.target_density)
        overflow_potential = (overflow ** 2).sum(dim=(1, 2, 3))
        
        # Boundary penalty
        boundary = self._boundary_penalty(positions, sizes, keep)
        
        potential = overflow_potential + self.boundary_weight * boundary
        
        return potential, density_smooth
    
    def _scatter_density(self, positions, sizes, keep):
        B, N, _ = positions.shape
        M = self.grid_size
        device = positions.device
        
        grid_x = (positions[..., 0] + 1) * 0.5 * (M - 1)
        grid_y = (positions[..., 1] + 1) * 0.5 * (M - 1)
        
        cell_area = (2.0 / M) ** 2
        macro_area = sizes[..., 0] * sizes[..., 1]
        macro_cells = macro_area / cell_area * keep.float()
        
        x0 = grid_x.floor().long().clamp(0, M - 1)
        y0 = grid_y.floor().long().clamp(0, M - 1)
        x1 = (x0 + 1).clamp(0, M - 1)
        y1 = (y0 + 1).clamp(0, M - 1)
        
        wx = grid_x - x0.float()
        wy = grid_y - y0.float()
        
        w00 = (1 - wx) * (1 - wy)
        w01 = (1 - wx) * wy
        w10 = wx * (1 - wy)
        w11 = wx * wy
        
        density = torch.zeros(B, M, M, device=device, dtype=positions.dtype)
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
        
        for wx_, x_, y_ in [(w00, x0, y0), (w01, x0, y1), (w10, x1, y0), (w11, x1, y1)]:
            flat_idx = batch_idx * M * M + y_ * M + x_
            density.view(-1).scatter_add_(0, flat_idx.view(-1), (wx_ * macro_cells).view(-1))
        
        return density.unsqueeze(1)
    
    def _boundary_penalty(self, positions, sizes, keep):
        half = sizes / 2
        min_ext = positions - half
        max_ext = positions + half
        below = F.relu(-1 - min_ext)
        above = F.relu(max_ext - 1)
        violation = (below + above) * keep.unsqueeze(-1).float()
        return (violation ** 2).sum(dim=(1, 2))


# ============================================================================
# FORCE B: WEIGHTED AVERAGE WIRELENGTH (WA-WL)
# ============================================================================

class DifferentiableHPWL(nn.Module):
    r"""
    Weighted Average Wirelength (WA-WL) - Smooth HPWL approximation.
    
    Theory:
    -------
    Standard HPWL uses max/min which are non-differentiable.
    WA-WL uses log-sum-exp (LSE) smooth approximation:
    
    .. math::
        x_{max} \approx \frac{1}{\gamma} \log \sum_i \exp(\gamma x_i)
        
        x_{min} \approx -\frac{1}{\gamma} \log \sum_i \exp(-\gamma x_i)
        
    The wirelength of a net:
    
    .. math::
        WL_{net} = (x_{max} - x_{min}) + (y_{max} - y_{min})
        
    Total wirelength:
    
    .. math::
        E_{WL} = \sum_{nets} WL_{net}
    
    Complexity: O(P) where P = total pins
    """
    
    def __init__(self, gamma: float = 10.0):
        """
        Args:
            gamma: LSE smoothing parameter. Higher = closer to true max/min.
        """
        super().__init__()
        self.gamma = gamma
    
    def forward(
        self,
        positions: torch.Tensor,
        net_to_pin: torch.Tensor,
        pin_to_macro: torch.Tensor,
        pin_offsets: torch.Tensor,
        rotation_onehot: Optional[torch.Tensor] = None,
        net_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute differentiable HPWL.
        
        Args:
            positions: (B, V, 2) macro positions
            net_to_pin: (num_nets, max_pins_per_net) pin indices, padded with -1
            pin_to_macro: (P,) macro index for each pin
            pin_offsets: (P, 2) pin offset from macro center
            rotation_onehot: (B, V, 4) optional rotation
            net_weights: (num_nets,) optional per-net weights
            
        Returns:
            hpwl: (B,) total wirelength per batch
        """
        B, V, _ = positions.shape
        device = positions.device
        num_nets, max_pins = net_to_pin.shape
        
        # Get pin positions
        macro_pos = positions[:, pin_to_macro, :]  # (B, P, 2)
        
        # Rotate pin offsets if needed
        if rotation_onehot is not None:
            rotated_offsets = self._rotate_offsets(pin_offsets, rotation_onehot, pin_to_macro)
        else:
            rotated_offsets = pin_offsets.unsqueeze(0).expand(B, -1, -1)
        
        pin_pos = macro_pos + rotated_offsets  # (B, P, 2)
        
        # Gather pin positions per net
        valid_mask = net_to_pin >= 0  # (num_nets, max_pins)
        safe_idx = net_to_pin.clamp(min=0)  # (num_nets, max_pins)
        
        # (B, num_nets, max_pins, 2)
        net_pin_pos = pin_pos[:, safe_idx, :]
        
        # Mask invalid pins with extreme values for LSE
        mask_expand = valid_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 2)
        
        # LSE max approximation
        masked_for_max = torch.where(mask_expand, net_pin_pos, torch.tensor(-1e9, device=device))
        lse_max = torch.logsumexp(self.gamma * masked_for_max, dim=2) / self.gamma  # (B, num_nets, 2)
        
        # LSE min approximation  
        masked_for_min = torch.where(mask_expand, net_pin_pos, torch.tensor(1e9, device=device))
        lse_min = -torch.logsumexp(-self.gamma * masked_for_min, dim=2) / self.gamma  # (B, num_nets, 2)
        
        # Wirelength per net
        wl_per_net = (lse_max - lse_min).sum(dim=-1)  # (B, num_nets)
        
        # Apply net weights
        if net_weights is not None:
            wl_per_net = wl_per_net * net_weights.unsqueeze(0)
        
        # Total
        hpwl = wl_per_net.sum(dim=1)  # (B,)
        
        return hpwl
    
    def _rotate_offsets(self, pin_offsets, rotation_onehot, pin_to_macro):
        """Apply rotation to pin offsets."""
        B = rotation_onehot.shape[0]
        P = pin_offsets.shape[0]
        device = pin_offsets.device
        
        # Rotation matrices
        R_all = torch.tensor([
            [[1, 0], [0, 1]],
            [[0, -1], [1, 0]],
            [[-1, 0], [0, -1]],
            [[0, 1], [-1, 0]],
        ], dtype=torch.float32, device=device)
        
        macro_rot = rotation_onehot[:, pin_to_macro, :]  # (B, P, 4)
        R = torch.einsum('bpk,kij->bpij', macro_rot, R_all)  # (B, P, 2, 2)
        rotated = torch.einsum('bpij,pj->bpi', R, pin_offsets)  # (B, P, 2)
        
        return rotated


# ============================================================================
# FORCE C: DIFFERENTIABLE RUDY (ROUTING CONGESTION)
# ============================================================================

class DifferentiableRUDY(nn.Module):
    r"""
    Differentiable RUDY (Rectangular Uniform Wire Density) for congestion.
    
    Theory:
    -------
    For each net, estimate routing demand as its bounding box area.
    Distribute this demand uniformly across the grid cells within the bbox.
    
    .. math::
        RUDY(i,j) = \sum_{nets} \frac{\mathbb{1}[(i,j) \in bbox_{net}]}{|bbox_{net}|}
        
    Congestion penalty when RUDY exceeds threshold:
    
    .. math::
        E_{congestion} = \sum_{i,j} \max(0, RUDY(i,j) - \tau)^2
        
    Implementation:
    ---------------
    Vectorized computation using 2D grid coordinate comparison.
    No for-loops over nets.
    
    Complexity: O(N_nets + M²)
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        threshold: float = 1.0,
        sigma: float = 1.5,
    ):
        """
        Args:
            grid_size: RUDY grid resolution
            threshold: Congestion threshold
            sigma: Gaussian smoothing for continuous gradients
        """
        super().__init__()
        self.grid_size = grid_size
        self.threshold = threshold
        self.sigma = sigma
        
        self.register_buffer('gaussian_kernel', self._make_kernel(sigma))
    
    def _make_kernel(self, sigma):
        ksize = int(4 * sigma) | 1
        ksize = max(ksize, 3)
        x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        g1d = torch.exp(-x**2 / (2 * sigma**2))
        g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
        g2d = g2d / g2d.sum()
        return g2d.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        positions: torch.Tensor,
        net_to_pin: torch.Tensor,
        pin_to_macro: torch.Tensor,
        pin_offsets: torch.Tensor,
        rotation_onehot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RUDY congestion penalty.
        
        Args:
            positions: (B, V, 2) macro positions
            net_to_pin: (num_nets, max_pins) pin indices
            pin_to_macro: (P,) macro for each pin
            pin_offsets: (P, 2) offsets
            rotation_onehot: (B, V, 4) optional
            
        Returns:
            congestion_penalty: (B,)
            rudy_map: (B, 1, M, M)
        """
        B, V, _ = positions.shape
        M = self.grid_size
        device = positions.device
        num_nets, max_pins = net_to_pin.shape
        
        # Get pin positions
        macro_pos = positions[:, pin_to_macro, :]  # (B, P, 2)
        
        if rotation_onehot is not None:
            R_all = torch.tensor([
                [[1, 0], [0, 1]], [[0, -1], [1, 0]],
                [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
            ], dtype=torch.float32, device=device)
            macro_rot = rotation_onehot[:, pin_to_macro, :]
            R = torch.einsum('bpk,kij->bpij', macro_rot, R_all)
            rotated = torch.einsum('bpij,pj->bpi', R, pin_offsets)
        else:
            rotated = pin_offsets.unsqueeze(0).expand(B, -1, -1)
        
        pin_pos = macro_pos + rotated  # (B, P, 2)
        
        # Compute net bounding boxes
        valid_mask = net_to_pin >= 0  # (num_nets, max_pins)
        safe_idx = net_to_pin.clamp(min=0)
        
        net_pin_pos = pin_pos[:, safe_idx, :]  # (B, num_nets, max_pins, 2)
        mask_expand = valid_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 2)
        
        masked_max = torch.where(mask_expand, net_pin_pos, torch.tensor(-1e9, device=device))
        masked_min = torch.where(mask_expand, net_pin_pos, torch.tensor(1e9, device=device))
        
        bbox_max = masked_max.max(dim=2).values  # (B, num_nets, 2)
        bbox_min = masked_min.min(dim=2).values  # (B, num_nets, 2)
        
        # Convert to grid coordinates
        bbox_max_grid = (bbox_max + 1) * 0.5 * (M - 1)  # (B, num_nets, 2)
        bbox_min_grid = (bbox_min + 1) * 0.5 * (M - 1)
        
        # RUDY accumulation via soft assignment
        rudy_map = self._accumulate_rudy(bbox_min_grid, bbox_max_grid, B, num_nets, M, device)
        
        # Smooth
        kernel = self.gaussian_kernel.to(device, rudy_map.dtype)
        padding = kernel.shape[-1] // 2
        rudy_smooth = F.conv2d(rudy_map, kernel, padding=padding)
        
        # Congestion penalty
        overflow = F.relu(rudy_smooth - self.threshold)
        penalty = (overflow ** 2).sum(dim=(1, 2, 3))
        
        return penalty, rudy_smooth
    
    def _accumulate_rudy(self, bbox_min, bbox_max, B, num_nets, M, device):
        """
        Vectorized RUDY accumulation using soft bbox indicators.
        
        Uses sigmoid to create soft rectangular regions for differentiability.
        """
        # Grid coordinates
        gx = torch.arange(M, device=device, dtype=torch.float32)
        gy = torch.arange(M, device=device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing='xy')  # (M, M)
        
        # Expand for broadcasting: (B, num_nets, M, M)
        grid_x = grid_x.view(1, 1, M, M).expand(B, num_nets, -1, -1)
        grid_y = grid_y.view(1, 1, M, M).expand(B, num_nets, -1, -1)
        
        # Soft indicators using sigmoid
        # x direction
        x_min = bbox_min[..., 0].view(B, num_nets, 1, 1)
        x_max = bbox_max[..., 0].view(B, num_nets, 1, 1)
        y_min = bbox_min[..., 1].view(B, num_nets, 1, 1)
        y_max = bbox_max[..., 1].view(B, num_nets, 1, 1)
        
        # Smooth step function
        k = 2.0  # Sharpness
        in_x = torch.sigmoid(k * (grid_x - x_min + 0.5)) * torch.sigmoid(k * (x_max - grid_x + 0.5))
        in_y = torch.sigmoid(k * (grid_y - y_min + 0.5)) * torch.sigmoid(k * (y_max - grid_y + 0.5))
        
        indicator = in_x * in_y  # (B, num_nets, M, M)
        
        # Normalize by bbox area
        bbox_size = (x_max - x_min + 1) * (y_max - y_min + 1)
        bbox_size = bbox_size.clamp(min=1.0)
        normalized = indicator / bbox_size
        
        # Sum over all nets
        rudy = normalized.sum(dim=1, keepdim=True)  # (B, 1, M, M)
        
        return rudy


# ============================================================================
# GRADIENT BALANCER
# ============================================================================

class GradientBalancer(nn.Module):
    r"""
    Multi-objective gradient balancing with adaptive weighting.
    
    Problem:
    --------
    Different objectives have vastly different gradient magnitudes:
    - Density gradient: ~1e2
    - HPWL gradient: ~1e0
    - RUDY gradient: ~1e1
    
    Solution:
    ---------
    1. GradNorm: Normalize gradients to same magnitude before combining
    2. Timestep-adaptive weighting: Different priorities at different stages
    
    .. math::
        g_{total} = \sum_i w_i(t) \cdot \frac{g_i}{\|g_i\| + \epsilon}
        
    Scheduling:
    - Early (t ≈ T): Prioritize HPWL (global structure)
    - Middle: Balance all objectives
    - Late (t → 0): Prioritize Density (legalization)
    """
    
    def __init__(
        self,
        density_weight: float = 1.0,
        hpwl_weight: float = 1.0,
        rudy_weight: float = 0.5,
        normalize_grads: bool = True,
        schedule: str = "cosine",  # "cosine", "linear", "constant"
    ):
        super().__init__()
        self.base_weights = {
            'density': density_weight,
            'hpwl': hpwl_weight,
            'rudy': rudy_weight,
        }
        self.normalize_grads = normalize_grads
        self.schedule = schedule
    
    def get_weights(self, t: int, T: int) -> Dict[str, float]:
        """
        Get timestep-adaptive weights.
        
        Args:
            t: Current timestep (T = max, 0 = final)
            T: Maximum timesteps
            
        Returns:
            weights: Dict of weight per objective
        """
        progress = 1.0 - t / T  # 0 at start, 1 at end
        
        if self.schedule == "cosine":
            # Cosine annealing
            alpha = 0.5 * (1 + math.cos(math.pi * progress))
        elif self.schedule == "linear":
            alpha = 1.0 - progress
        else:
            alpha = 0.5
        
        # Early: more HPWL. Late: more Density
        weights = {
            'density': self.base_weights['density'] * (0.3 + 0.7 * progress),
            'hpwl': self.base_weights['hpwl'] * (1.0 - 0.5 * progress),
            'rudy': self.base_weights['rudy'] * (0.5 + 0.5 * progress),
        }
        
        return weights
    
    def combine_gradients(
        self,
        grads: Dict[str, torch.Tensor],
        t: int,
        T: int,
    ) -> torch.Tensor:
        """
        Combine gradients with normalization and weighting.
        
        Args:
            grads: Dict of gradient tensors, each (B, V, 2)
            t: Current timestep
            T: Max timesteps
            
        Returns:
            combined: (B, V, 2) combined gradient
        """
        weights = self.get_weights(t, T)
        combined = None
        
        for name, g in grads.items():
            if g is None:
                continue
                
            w = weights.get(name, 1.0)
            
            if self.normalize_grads:
                # Per-sample normalization
                g_norm = g.norm(dim=(1, 2), keepdim=True) + 1e-8
                g = g / g_norm
            
            weighted = w * g
            
            if combined is None:
                combined = weighted
            else:
                combined = combined + weighted
        
        return combined if combined is not None else torch.zeros_like(list(grads.values())[0])


# ============================================================================
# TRINITY GUIDANCE ENGINE
# ============================================================================

class TrinityGuidance(nn.Module):
    """
    Unified multi-objective guidance for DiffPlace.
    
    Combines:
    - Electrostatic Density (spreading)
    - WA-WL Wirelength (clustering)
    - RUDY Congestion (routing)
    
    With gradient balancing for stable optimization.
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        density_sigma: float = 2.0,
        target_density: float = 1.0,
        hpwl_gamma: float = 10.0,
        rudy_threshold: float = 1.0,
        density_weight: float = 1.0,
        hpwl_weight: float = 1.0,
        rudy_weight: float = 0.5,
        boundary_weight: float = 10.0,
        normalize_grads: bool = True,
    ):
        super().__init__()
        
        # Force A: Density
        self.density = ElectrostaticDensity(
            grid_size=grid_size,
            sigma=density_sigma,
            target_density=target_density,
            boundary_weight=boundary_weight,
        )
        
        # Force B: WA-WL
        self.hpwl = DifferentiableHPWL(gamma=hpwl_gamma)
        
        # Force C: RUDY
        self.rudy = DifferentiableRUDY(
            grid_size=grid_size,
            threshold=rudy_threshold,
        )
        
        # Gradient balancer
        self.balancer = GradientBalancer(
            density_weight=density_weight,
            hpwl_weight=hpwl_weight,
            rudy_weight=rudy_weight,
            normalize_grads=normalize_grads,
        )
    
    def compute_potentials(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        net_to_pin: torch.Tensor,
        pin_to_macro: torch.Tensor,
        pin_offsets: torch.Tensor,
        rotation_onehot: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all objective potentials.
        
        Returns:
            potentials: Dict with 'density', 'hpwl', 'rudy' scalars (B,)
        """
        # Density
        density_pot, density_map = self.density(positions, sizes, mask)
        
        # HPWL
        hpwl_pot = self.hpwl(positions, net_to_pin, pin_to_macro, pin_offsets, rotation_onehot)
        
        # RUDY
        rudy_pot, rudy_map = self.rudy(positions, net_to_pin, pin_to_macro, pin_offsets, rotation_onehot)
        
        return {
            'density': density_pot,
            'hpwl': hpwl_pot,
            'rudy': rudy_pot,
            'density_map': density_map,
            'rudy_map': rudy_map,
        }
    
    def compute_guidance_force(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        net_to_pin: torch.Tensor,
        pin_to_macro: torch.Tensor,
        pin_offsets: torch.Tensor,
        rotation_onehot: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        t: int = 0,
        T: int = 1000,
    ) -> torch.Tensor:
        """
        Compute combined guidance force (negative gradient).
        
        Returns:
            force: (B, V, 2) guidance force to apply to positions
        """
        positions = positions.detach().requires_grad_(True)
        
        potentials = self.compute_potentials(
            positions, sizes, net_to_pin, pin_to_macro, pin_offsets, 
            rotation_onehot, mask
        )
        
        grads = {}
        
        # Compute gradients
        for name in ['density', 'hpwl', 'rudy']:
            pot = potentials[name]
            grad = torch.autograd.grad(
                pot.sum(), positions, 
                create_graph=False, retain_graph=True
            )[0]
            grads[name] = grad
        
        # Combine with balancing
        combined = self.balancer.combine_gradients(grads, t, T)
        
        # Force is negative gradient (move downhill)
        force = -combined
        
        return force
