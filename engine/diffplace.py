"""
DiffPlace Model

Unified architecture integrating:
- VectorGNNV2Global backbone (Phase 3)
- Trinity Guidance Engine (Density + HPWL + RUDY)
- Discrete rotation via Gumbel-Softmax (Phase 1)
- Scalable O(N) operations throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

from engine.networks.vector_gnn import VectorGNNV2Global, VectorGNNV2GlobalLarge, DiscreteRotationHead
from engine.diffusion.trinity_guidance import TrinityGuidance, GradientBalancer
from engine.diffusion.overlap_loss import OverlapLoss


class DiffPlace(nn.Module):
    """
    DiffPlace State-of-the-Art Unified Model.
    
    Architecture:
    -------------
    1. Backbone: VectorGNNV2Global with bifurcated heads
       - Position head: continuous noise prediction
       - Rotation head: 4-class Gumbel-Softmax
       
    2. Diffusion Process:
       - Position: Gaussian diffusion (DDPM)
       - Rotation: Clean conditioning (no diffusion on discrete)
       
    3. Guidance: Trinity Engine
       - Force A: Electrostatic Density (spreading)
       - Force B: WA-WL (wirelength minimization)
       - Force C: RUDY (congestion avoidance)
       
    4. Gradient Balancing:
       - GradNorm for magnitude equalization
       - Timestep-adaptive weighting
    """
    
    def __init__(
        self,
        # Backbone params
        hidden_size: int = 128,
        t_encoding_dim: int = 64,
        cond_node_features: int = 2,
        edge_features: int = 4,
        num_blocks: int = 4,
        layers_per_block: int = 2,
        num_heads: int = 4,
        num_rotations: int = 4,
        rotation_temperature: float = 1.0,
        global_context_every: int = 1,
        mask_key: str = "is_ports",
        # Diffusion params
        max_diffusion_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        # Guidance params
        guidance_grid_size: int = 64,
        density_sigma: float = 2.0,
        target_density: float = 1.0,
        hpwl_gamma: float = 10.0,
        rudy_threshold: float = 1.0,
        density_weight: float = 1.0,
        hpwl_weight: float = 1.0,
        rudy_weight: float = 0.5,
        boundary_weight: float = 10.0,
        guidance_start_step: float = 0.7,
        guidance_strength: float = 1.0,
        # Device
        device: str = "cpu",
        # Memory optimization
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.device = device
        self.max_diffusion_steps = max_diffusion_steps
        self.guidance_start_step = guidance_start_step
        self.guidance_strength = guidance_strength
        self.mask_key = mask_key
        self._gradient_checkpointing = gradient_checkpointing
        
        # === Backbone ===
        self.backbone = VectorGNNV2Global(
            hidden_size=hidden_size,
            t_encoding_dim=t_encoding_dim,
            cond_node_features=cond_node_features,
            edge_features=edge_features,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            num_heads=num_heads,
            num_rotations=num_rotations,
            rotation_temperature=rotation_temperature,
            global_context_every=global_context_every,
            mask_key=mask_key,
            device=device,
        )
        
        # === Time embedding ===
        half_dim = t_encoding_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb_freqs = torch.exp(torch.arange(half_dim) * -emb_scale)
        self.register_buffer('time_freqs', emb_freqs)
        self.t_encoding_dim = t_encoding_dim
        
        # === Diffusion schedule ===
        betas = torch.linspace(beta_start, beta_end, max_diffusion_steps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - alpha_bar))
        self.register_buffer('sigma', torch.sqrt(betas))
        
        # === Trinity Guidance ===
        self.guidance = TrinityGuidance(
            grid_size=guidance_grid_size,
            density_sigma=density_sigma,
            target_density=target_density,
            hpwl_gamma=hpwl_gamma,
            rudy_threshold=rudy_threshold,
            density_weight=density_weight,
            hpwl_weight=hpwl_weight,
            rudy_weight=rudy_weight,
            boundary_weight=boundary_weight,
        )
        
        # Loss function
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce VRAM at cost of compute."""
        self._gradient_checkpointing = True
        # Enable for backbone blocks
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks:
                block.gradient_checkpointing = True
        # print("Gradient checkpointing ENABLED")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks:
                block.gradient_checkpointing = False
        # print("Gradient checkpointing DISABLED")
    
    def _init_time_encoding(self, T, dim):
        """Initialize sinusoidal time encoding table."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        
        t = torch.arange(T).float()
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        self.register_buffer('time_encoding_table', emb)
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Get time embedding for timesteps using sinusoidal encoding."""
        # t: (B,) integer timesteps
        t_float = t.float().unsqueeze(-1)  # (B, 1)
        freqs = self.time_freqs.unsqueeze(0)  # (1, half_dim)
        emb = t_float * freqs  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, t_encoding_dim)
        return emb
    
    def forward_noising(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: add noise.
        
        Args:
            x_0: (B, V, 2) clean positions
            t: (B,) timesteps
            
        Returns:
            x_t: (B, V, 2) noisy positions
            epsilon: (B, V, 2) noise that was added
        """
        B = x_0.shape[0]
        coef_shape = (B, 1, 1)
        
        epsilon = torch.randn_like(x_0)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(*coef_shape)
        sqrt_one_minus_t = self.sqrt_one_minus_alpha_bar[t].view(*coef_shape)
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_t * epsilon
        
        return x_t, epsilon
    
    def forward(
        self,
        x_t: torch.Tensor,
        cond,
        t: torch.Tensor,
        rotation_temperature: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict noise and rotation.
        
        Args:
            x_t: (B, V, 2) noisy positions
            cond: PyG Data object
            t: (B,) timesteps
            rotation_temperature: optional Gumbel-Softmax temperature
            
        Returns:
            eps_pred: (B, V, 2) predicted noise
            rot_onehot: (B, V, 4) rotation prediction
            rot_logits: (B, V, 4) rotation logits
        """
        t_embed = self.get_time_embedding(t)
        return self.backbone(x_t, cond, t_embed, rotation_temperature)
    
    def loss(
        self,
        x_0: torch.Tensor,
        rot_true: torch.Tensor,
        cond,
        rotation_loss_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x_0: (B, V, 2) clean positions
            rot_true: (B, V) true rotation indices (0-3)
            cond: PyG Data object
            rotation_loss_weight: weight for rotation CE loss
            
        Returns:
            losses: Dict with 'total', 'position', 'rotation'
        """
        B, V, _ = x_0.shape
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(1, self.max_diffusion_steps + 1, (B,), device=device)
        
        # Forward diffusion
        x_t, epsilon = self.forward_noising(x_0, t - 1)  # 0-indexed
        
        # Get mask
        mask = None
        if self.mask_key and hasattr(cond, self.mask_key):
            mask = getattr(cond, self.mask_key)  # (V,)
            mask = mask.unsqueeze(0).expand(B, -1)  # (B, V)
        
        # Apply mask to x_t (keep original for masked nodes)
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).expand(-1, -1, 2)
            x_t = torch.where(mask_3d, x_0, x_t)
        
        # Predict
        eps_pred, rot_onehot, rot_logits = self(x_t, cond, t)
        
        # Position loss (MSE on noise)
        loss_pos = self.mse_loss(eps_pred, epsilon)
        if mask is not None:
            loss_pos = loss_pos * (~mask).float().unsqueeze(-1)
        loss_pos = loss_pos.mean()
        
        # Rotation loss (CrossEntropy)
        loss_rot = self.ce_loss(rot_logits.view(-1, 4), rot_true.view(-1))
        if mask is not None:
            loss_rot = loss_rot.view(B, V) * (~mask).float()
        loss_rot = loss_rot.mean()
        
        # Total
        loss_total = loss_pos + rotation_loss_weight * loss_rot
        
        return {
            'total': loss_total,
            'position': loss_pos,
            'rotation': loss_rot,
        }
    
    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size: int = 1,
        guidance_scale: float = 1.0,
        overlap_guidance: bool = False,
        overlap_guidance_scale_max: float = 1000.0,
        overlap_guidance_power: float = 2.0,
        overlap_sizes: Optional[torch.Tensor] = None,
        overlap_guidance_start_frac: float = 0.0,
        overlap_guidance_grad_norm: Optional[float] = None,
        overlap_guidance_grad_clip: Optional[float] = None,
        return_intermediates: bool = False,
        intermediate_every: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Sample placements using reverse diffusion with Trinity guidance.
        
        Args:
            cond: PyG Data object with graph structure
            batch_size: number of samples
            guidance_scale: multiplier for guidance force
            return_intermediates: whether to save intermediate states
            intermediate_every: save every N steps
            
        Returns:
            positions: (B, V, 2) final positions
            rotations: (B, V, 4) final rotation one-hot
            intermediates: list of (positions, rotations) if requested
        """
        B = batch_size
        V = cond.x.shape[0]
        device = next(self.parameters()).device
        T = self.max_diffusion_steps
        
        # Get mask
        mask = None
        if self.mask_key and hasattr(cond, self.mask_key):
            mask = getattr(cond, self.mask_key)
        
        # Initial noise
        x = torch.randn(B, V, 2, device=device)
        
        # Apply mask (keep fixed positions for ports)
        if mask is not None and hasattr(cond, 'pos'):
            fixed_pos = cond.pos.unsqueeze(0).expand(B, -1, -1)[:, :, :2]
            mask_3d = mask.unsqueeze(0).unsqueeze(-1).expand(B, V, 2)
            x = torch.where(mask_3d, fixed_pos, x)
        
        intermediates = []
        
        overlap_loss_fn = OverlapLoss().to(device)

        # Reverse diffusion
        for t in range(T, 0, -1):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise and rotation
            eps_pred, rot_onehot, rot_logits = self(x, cond, t_tensor)
            
            # === Gradient-Guided Legalization (overlap forbiddance) ===
            if overlap_guidance:
                # Quadratic schedule: small early, large late
                frac = 1.0 - (float(t) / float(T))
                if frac < float(overlap_guidance_start_frac):
                    lambda_t = 0.0
                else:
                    lambda_t = float(overlap_guidance_scale_max) * (frac ** float(overlap_guidance_power))

                if lambda_t > 0:
                    macro_mask = None
                    if hasattr(cond, self.mask_key):
                        # Ports are fixed; guide only movable nodes
                        macro_mask = ~getattr(cond, self.mask_key)

                    with torch.enable_grad():
                        x_grad = x.detach().requires_grad_(True)
                        sizes = overlap_sizes
                        if sizes is None and hasattr(cond, "overlap_sizes"):
                            sizes = getattr(cond, "overlap_sizes")
                        if sizes is None:
                            sizes = cond.x
                        if rot_onehot is not None:
                            sizes = DiscreteRotationHead.compute_effective_size(
                                sizes, rot_onehot.detach()
                            )
                        loss_ov = overlap_loss_fn(x_grad, sizes, macro_mask=macro_mask)
                        g = torch.autograd.grad(loss_ov, x_grad, retain_graph=False, create_graph=False)[0]
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).detach()
                    if overlap_guidance_grad_norm is not None or overlap_guidance_grad_clip is not None:
                        eps = 1e-12
                        gn = torch.linalg.vector_norm(g, ord=2, dim=-1, keepdim=True)
                        if overlap_guidance_grad_norm is not None:
                            g = g * (float(overlap_guidance_grad_norm) / (gn + eps))
                        if overlap_guidance_grad_clip is not None:
                            clip = float(overlap_guidance_grad_clip)
                            scale = torch.clamp(clip / (gn + eps), max=1.0)
                            g = g * scale
                    eps_pred = eps_pred - (lambda_t * g)

            # DDPM update
            alpha_t = self.alphas[t - 1]
            alpha_bar_t = self.alpha_bar[t - 1]
            sqrt_alpha_bar_t = self.sqrt_alpha_bar[t - 1]
            sqrt_one_minus_t = self.sqrt_one_minus_alpha_bar[t - 1]
            
            # Predicted x_0
            x_0_pred = (x - sqrt_one_minus_t * eps_pred) / sqrt_alpha_bar_t
            x_0_pred = torch.clamp(x_0_pred, -2, 2)  # Stability clamp
            
            # Compute posterior mean
            mu = (1.0 / torch.sqrt(alpha_t)) * (
                x - (self.betas[t - 1] / sqrt_one_minus_t) * eps_pred
            )
            
            # Add noise if not last step
            if t > 1:
                noise = torch.randn_like(x)
                sigma_t = self.sigma[t - 1]
                x = mu + sigma_t * noise
            else:
                x = mu
            
            # Apply guidance after warmup
            guidance_threshold = int(self.guidance_start_step * T)
            if t <= guidance_threshold and hasattr(cond, 'net_to_pin'):
                force = self._compute_guidance_force(
                    x, cond, rot_onehot, t, T
                )
                x = x + guidance_scale * self.guidance_strength * force
            
            # Apply mask
            if mask is not None and hasattr(cond, 'pos'):
                x = torch.where(mask_3d, fixed_pos, x)
            
            # Save intermediate
            if return_intermediates and t % intermediate_every == 0:
                intermediates.append((x.clone(), rot_onehot.clone()))
        
        # Final rotation (hard argmax)
        with torch.no_grad():
            _, rot_onehot, _ = self(x, cond, torch.ones(B, device=device, dtype=torch.long))
        
        return x, rot_onehot, intermediates
    
    @torch.no_grad()
    def sample_ddim(
        self,
        cond,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        overlap_guidance: bool = False,
        overlap_guidance_scale_max: float = 1000.0,
        overlap_guidance_power: float = 2.0,
        overlap_sizes: Optional[torch.Tensor] = None,
        overlap_guidance_start_frac: float = 0.0,
        overlap_guidance_grad_norm: Optional[float] = None,
        overlap_guidance_grad_clip: Optional[float] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        DDIM sampling for faster inference (50 steps vs 1000).
        
        DDIM (Denoising Diffusion Implicit Models) uses a non-Markovian
        process that allows skipping steps while maintaining quality.
        
        Args:
            cond: PyG Data object
            batch_size: number of samples
            num_inference_steps: number of denoising steps (default 50)
            eta: controls stochasticity (0 = deterministic, 1 = DDPM)
            guidance_scale: multiplier for guidance force
            return_intermediates: save intermediate states
            
        Returns:
            positions, rotations, intermediates
        """
        B = batch_size
        V = cond.x.shape[0]
        device = next(self.parameters()).device
        T = self.max_diffusion_steps
        
        # Create timestep schedule (uniform spacing)
        timesteps = torch.linspace(T, 1, num_inference_steps, dtype=torch.long, device=device)
        
        # Get mask
        mask = None
        mask_3d = None
        fixed_pos = None
        if self.mask_key and hasattr(cond, self.mask_key):
            mask = getattr(cond, self.mask_key)
        
        # Initial noise
        x = torch.randn(B, V, 2, device=device)
        
        if mask is not None and hasattr(cond, 'pos'):
            fixed_pos = cond.pos.unsqueeze(0).expand(B, -1, -1)[:, :, :2]
            mask_3d = mask.unsqueeze(0).unsqueeze(-1).expand(B, V, 2)
            x = torch.where(mask_3d, fixed_pos, x)
        
        intermediates = []

        overlap_loss_fn = OverlapLoss().to(device)
        
        # DDIM reverse process
        for i, t in enumerate(timesteps):
            t_tensor = t.expand(B)
            t_idx = t.item() - 1  # 0-indexed
            
            # Get next timestep (or 0 for last step)
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1].item()
                t_next_idx = t_next - 1
            else:
                t_next = 0
                t_next_idx = -1
            
            # Predict noise
            eps_pred, rot_onehot, _ = self(x, cond, t_tensor)

            # === Gradient-Guided Legalization (overlap forbiddance) ===
            if overlap_guidance:
                # Quadratic schedule: small early, large late
                frac = 1.0 - (float(t.item()) / float(T))
                if frac < float(overlap_guidance_start_frac):
                    lambda_t = 0.0
                else:
                    lambda_t = float(overlap_guidance_scale_max) * (frac ** float(overlap_guidance_power))

                if lambda_t > 0:
                    macro_mask = None
                    if hasattr(cond, self.mask_key):
                        macro_mask = ~getattr(cond, self.mask_key)

                    with torch.enable_grad():
                        x_grad = x.detach().requires_grad_(True)
                        sizes = overlap_sizes
                        if sizes is None and hasattr(cond, "overlap_sizes"):
                            sizes = getattr(cond, "overlap_sizes")
                        if sizes is None:
                            sizes = cond.x
                        if rot_onehot is not None:
                            sizes = DiscreteRotationHead.compute_effective_size(
                                sizes, rot_onehot.detach()
                            )
                        loss_ov = overlap_loss_fn(x_grad, sizes, macro_mask=macro_mask)
                        g = torch.autograd.grad(loss_ov, x_grad, retain_graph=False, create_graph=False)[0]
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).detach()
                    if overlap_guidance_grad_norm is not None or overlap_guidance_grad_clip is not None:
                        eps = 1e-12
                        gn = torch.linalg.vector_norm(g, ord=2, dim=-1, keepdim=True)
                        if overlap_guidance_grad_norm is not None:
                            g = g * (float(overlap_guidance_grad_norm) / (gn + eps))
                        if overlap_guidance_grad_clip is not None:
                            clip = float(overlap_guidance_grad_clip)
                            scale = torch.clamp(clip / (gn + eps), max=1.0)
                            g = g * scale
                    eps_pred = eps_pred - (lambda_t * g)
            
            # Current alpha values
            alpha_bar_t = self.alpha_bar[t_idx]
            sqrt_alpha_bar_t = self.sqrt_alpha_bar[t_idx]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_idx]
            
            # Next alpha values
            if t_next > 0:
                alpha_bar_t_next = self.alpha_bar[t_next_idx]
                sqrt_alpha_bar_t_next = torch.sqrt(alpha_bar_t_next)
                sqrt_one_minus_alpha_bar_t_next = torch.sqrt(1 - alpha_bar_t_next)
            else:
                alpha_bar_t_next = torch.tensor(1.0, device=device)
                sqrt_alpha_bar_t_next = torch.tensor(1.0, device=device)
                sqrt_one_minus_alpha_bar_t_next = torch.tensor(0.0, device=device)
            
            # Predicted x_0
            x_0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
            x_0_pred = torch.clamp(x_0_pred, -2, 2)
            
            # DDIM formula
            # x_{t-1} = sqrt(α_{t-1}) * x_0_pred + sqrt(1 - α_{t-1} - σ²) * ε_pred + σ * noise
            sigma = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_next)
            
            # Direction pointing to x_t
            pred_dir = sqrt_one_minus_alpha_bar_t_next * eps_pred
            if eta > 0:
                # Adjust for sigma
                pred_dir = torch.sqrt(1 - alpha_bar_t_next - sigma**2) * eps_pred
            
            # Compute x_{t-1}
            x = sqrt_alpha_bar_t_next * x_0_pred + pred_dir
            
            # Add noise if eta > 0 and not last step
            if eta > 0 and t_next > 0:
                noise = torch.randn_like(x)
                x = x + sigma * noise
            
            # Apply guidance
            guidance_threshold = int(self.guidance_start_step * T)
            if t.item() <= guidance_threshold and hasattr(cond, 'net_to_pin'):
                # Scale guidance for DDIM (multiply by sqrt(1-alpha))
                guidance_strength_scaled = self.guidance_strength * sqrt_one_minus_alpha_bar_t.item()
                force = self._compute_guidance_force(x, cond, rot_onehot, t.item(), T)
                x = x + guidance_scale * guidance_strength_scaled * force
            
            # Apply mask
            if mask_3d is not None and fixed_pos is not None:
                x = torch.where(mask_3d, fixed_pos, x)
            
            # Save intermediate
            if return_intermediates:
                intermediates.append((x.clone(), rot_onehot.clone()))
        
        # Final rotation
        _, rot_onehot, _ = self(x, cond, torch.ones(B, device=device, dtype=torch.long))
        
        return x, rot_onehot, intermediates
    
    def _compute_guidance_force(
        self,
        positions: torch.Tensor,
        cond,
        rotation_onehot: torch.Tensor,
        t: int,
        T: int,
    ) -> torch.Tensor:
        """Compute Trinity guidance force. Enables grad even inside no_grad context."""
        # Get sizes with rotation
        sizes = cond.x  # (V, 2)
        eff_sizes = DiscreteRotationHead.compute_effective_size(sizes, rotation_onehot)
        
        # Mask
        mask = None
        if self.mask_key and hasattr(cond, self.mask_key):
            mask = getattr(cond, self.mask_key)
        
        # Net structure
        net_to_pin = cond.net_to_pin if hasattr(cond, 'net_to_pin') else None
        pin_to_macro = cond.pin_to_macro if hasattr(cond, 'pin_to_macro') else None
        pin_offsets = cond.pin_offsets if hasattr(cond, 'pin_offsets') else None
        
        if net_to_pin is None:
            return torch.zeros_like(positions)
        
        # Enable gradient computation for guidance
        with torch.enable_grad():
            pos_grad = positions.detach().requires_grad_(True)
            force = self.guidance.compute_guidance_force(
                pos_grad, eff_sizes, 
                net_to_pin, pin_to_macro, pin_offsets,
                rotation_onehot.detach() if rotation_onehot is not None else None, 
                mask, t, T
            )
        
        return force.detach()


class DiffPlaceLarge(DiffPlace):
    """Larger variant for complex designs."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_size', 256)
        kwargs.setdefault('num_blocks', 5)
        kwargs.setdefault('layers_per_block', 3)
        kwargs.setdefault('num_heads', 8)
        kwargs.setdefault('guidance_grid_size', 128)
        super().__init__(**kwargs)
