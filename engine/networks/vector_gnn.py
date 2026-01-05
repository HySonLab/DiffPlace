import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEncoder(nn.Module):
    """Encodes 2D positions into higher-dimensional representations."""
    
    def __init__(self, dim, max_freq=10.0):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D encoding"
        
        half_dim = dim // 4
        emb = math.log(max_freq) / (half_dim - 1) if half_dim > 1 else 0
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('freqs', emb)

    def forward(self, pos):
        # pos: (..., 2)
        # output: (..., dim)
        x = pos[..., 0:1]  # (..., 1)
        y = pos[..., 1:2]  # (..., 1)
        
        x_emb = x * self.freqs  # (..., dim//4)
        y_emb = y * self.freqs  # (..., dim//4)
        
        emb = torch.cat([
            torch.sin(x_emb), torch.cos(x_emb),
            torch.sin(y_emb), torch.cos(y_emb)
        ], dim=-1)  # (..., dim)
        
        return emb


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for discrete timesteps."""
    
    def __init__(self, dim, max_timesteps=1000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, t):
        # t: (B,) integer timesteps
        t = t.float()
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for time conditioning.
    
    Applies affine transformation: (1 + gamma) * x + beta
    where gamma and beta are predicted from conditioning signal.
    Adding 1 ensures identity initialization when gamma output is 0.
    """
    
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feature_dim)
        self.beta_proj = nn.Linear(cond_dim, feature_dim)
        
        # Initialize so that gamma outputs 0 (effective scale=1) and beta=0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x, cond):
        # x: (B, V, D) or (B, D)
        # cond: (B, cond_dim)
        gamma = self.gamma_proj(cond)  # (B, D)
        beta = self.beta_proj(cond)    # (B, D)
        
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (B, 1, D)
            beta = beta.unsqueeze(1)    # (B, 1, D)
        
        # (1 + gamma) ensures scale starts at 1
        return (1 + gamma) * x + beta


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm and SiLU activation."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VectorMessagePassingLayer(nn.Module):
    """Vector-wise message passing layer with relative position encoding."""
    
    def __init__(
        self,
        hidden_dim,
        pos_encoding_dim=32,
        num_heads=4,
        dropout=0.0,
        use_edge_attr=False,
        edge_dim=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_encoding_dim = pos_encoding_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_attr = use_edge_attr

        self.pos_encoder = SinusoidalPosEncoder(pos_encoding_dim)

        # Message MLP: h_j || h_i || rel_pos_enc [|| edge_attr]
        msg_in_dim = 2 * hidden_dim + pos_encoding_dim
        if use_edge_attr:
            msg_in_dim += edge_dim
        
        self.message_mlp = MLP(
            in_dim=msg_in_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Attention weights
        self.attn_proj = nn.Linear(msg_in_dim, num_heads)

        # Update MLP
        self.update_mlp = MLP(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, edge_index, edge_attr=None, batch_size=None):
        """
        Forward pass with batch handling.
        
        Args:
            x: (B, V, D) node features
            pos: (B, V, 2) node positions
            edge_index: (2, E) edge indices (shared across batch)
            edge_attr: (E, edge_dim) optional edge attributes
            batch_size: int, inferred from x if None
        
        Returns:
            x_out: (B, V, D) updated node features
        """
        B, V, D = x.shape
        E = edge_index.shape[1]

        # Get source and target indices
        src, dst = edge_index[0], edge_index[1]

        # Compute relative positions for each batch
        pos_src = pos[:, src, :]  # (B, E, 2)
        pos_dst = pos[:, dst, :]  # (B, E, 2)
        rel_pos = pos_src - pos_dst  # (B, E, 2)

        # Encode relative positions
        rel_pos_enc = self.pos_encoder(rel_pos)  # (B, E, pos_encoding_dim)

        # Gather node features
        x_src = x[:, src, :]  # (B, E, D)
        x_dst = x[:, dst, :]  # (B, E, D)

        # Concatenate for message computation
        msg_input = torch.cat([x_src, x_dst, rel_pos_enc], dim=-1)
        
        if self.use_edge_attr and edge_attr is not None:
            # Expand edge_attr for batch: (E, edge_dim) -> (B, E, edge_dim)
            edge_attr_expanded = edge_attr.unsqueeze(0).expand(B, -1, -1)
            msg_input = torch.cat([msg_input, edge_attr_expanded], dim=-1)

        # Compute messages
        messages = self.message_mlp(msg_input)  # (B, E, D)

        # Compute attention weights
        attn_logits = self.attn_proj(msg_input)  # (B, E, num_heads)
        
        # Aggregate messages with attention
        x_out = self._aggregate_with_attention(
            messages, attn_logits, dst, V, B
        )  # (B, V, D)

        # Residual connection + update
        x_combined = torch.cat([x, x_out], dim=-1)  # (B, V, 2*D)
        x_update = self.update_mlp(x_combined)  # (B, V, D)
        
        x_out = self.layer_norm(x + self.dropout(x_update))
        
        return x_out

    def _aggregate_with_attention(self, messages, attn_logits, dst, num_nodes, batch_size):
        B, E, D = messages.shape
        
        # Reshape for multi-head
        messages_heads = messages.view(B, E, self.num_heads, self.head_dim)
        
        # Compute attention weights per head
        attn_weights = self._scatter_softmax(attn_logits, dst, num_nodes)  # (B, E, num_heads)
        
        # Weight messages
        weighted_messages = attn_weights.unsqueeze(-1) * messages_heads  # (B, E, num_heads, head_dim)
        weighted_messages = weighted_messages.view(B, E, D)  # (B, E, D)

        # Scatter add to aggregate
        aggregated = self._scatter_add(weighted_messages, dst, num_nodes)  # (B, V, D)
        
        return aggregated

    def _scatter_softmax(self, src, index, num_nodes):
        B, E, H = src.shape
        
        # Compute max per node for numerical stability
        max_vals = torch.full((B, num_nodes, H), float('-inf'), device=src.device, dtype=src.dtype)
        index_expanded = index.view(1, E, 1).expand(B, E, H)
        max_vals = max_vals.scatter_reduce(1, index_expanded, src, reduce='amax', include_self=False)
        max_vals = max_vals.gather(1, index_expanded)  # (B, E, H)
        
        # Compute exp(src - max)
        exp_src = torch.exp(src - max_vals)
        
        # Sum per node
        sum_exp = torch.zeros((B, num_nodes, H), device=src.device, dtype=src.dtype)
        sum_exp = sum_exp.scatter_add(1, index_expanded, exp_src)
        sum_exp = sum_exp.gather(1, index_expanded)  # (B, E, H)
        
        # Normalize
        softmax_out = exp_src / (sum_exp + 1e-8)
        
        return softmax_out

    def _scatter_add(self, src, index, num_nodes):
        B, E, D = src.shape
        out = torch.zeros((B, num_nodes, D), device=src.device, dtype=src.dtype)
        index_expanded = index.view(1, E, 1).expand(B, E, D)
        out = out.scatter_add(1, index_expanded, src)
        return out


class VectorGNNBlock(nn.Module):
    """GNN Block with FiLM time conditioning at each layer."""
    
    def __init__(
        self,
        hidden_dim,
        t_cond_dim,
        pos_encoding_dim=32,
        num_heads=4,
        num_layers=2,
        ff_dim=None,
        dropout=0.0,
        use_edge_attr=False,
        edge_dim=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        if ff_dim is None:
            ff_dim = hidden_dim * 4

        # Message passing layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                VectorMessagePassingLayer(
                    hidden_dim=hidden_dim,
                    pos_encoding_dim=pos_encoding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_edge_attr=use_edge_attr,
                    edge_dim=edge_dim,
                )
            )

        # FiLM conditioning for each layer
        self.film_layers = nn.ModuleList([
            FiLM(t_cond_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        
        # FiLM for feed-forward output
        self.ff_film = FiLM(t_cond_dim, hidden_dim)

    def forward(self, x, pos, edge_index, t_cond, edge_attr=None):
        """
        Args:
            x: (B, V, D) node features
            pos: (B, V, 2) node positions
            edge_index: (2, E) edge indices
            t_cond: (B, t_cond_dim) time conditioning vector
            edge_attr: (E, edge_dim) optional
        
        Returns:
            x: (B, V, D) updated features
        """
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, pos, edge_index, edge_attr)
            x = film(x, t_cond)  # Apply FiLM after each message passing
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        ff_out = self.ff_film(ff_out, t_cond)  # Apply FiLM to FF output
        x = self.ff_norm(x + ff_out)
        
        return x


class InputProjection(nn.Module):
    """Projects input features into hidden space.
    
    Handles:
    - Position (x, y) → Sinusoidal encoding + Linear projection
    - Macro features (w, h, type, ...) → Embedding
    - Mask features → Direct embedding
    """
    
    def __init__(
        self,
        in_node_features,      # dimension of input x (typically 2 for positions)
        cond_node_features,    # dimension of conditioning node features (w, h, ...)
        hidden_dim,
        pos_encoding_dim=32,
        input_encoding_dim=32,
        mask_key=None,
    ):
        super().__init__()
        self.mask_key = mask_key
        self.pos_encoding_dim = pos_encoding_dim
        self.input_encoding_dim = input_encoding_dim
        
        # Sinusoidal encoder for input positions
        self.pos_encoder = SinusoidalPosEncoder(pos_encoding_dim)
        
        # Additional sinusoidal encoding with learnable frequencies
        if input_encoding_dim > 0:
            MAX_FREQ = 100
            self.input_encoding_freqs = nn.Parameter(
                torch.exp(
                    math.log(MAX_FREQ) * 
                    torch.arange(0, input_encoding_dim // 2, dtype=torch.float32) / 
                    (input_encoding_dim // 2)
                ).view(1, 1, 1, input_encoding_dim // 2),
                requires_grad=False
            )
            spatial_enc_dim = (in_node_features + cond_node_features) * input_encoding_dim
            self.encoding_proj = nn.Linear(spatial_enc_dim, hidden_dim)
        else:
            self.encoding_proj = None
        
        # Input dimension: pos_encoding + raw_features + cond + (optional mask)
        proj_in_dim = pos_encoding_dim + in_node_features + cond_node_features
        if mask_key is not None:
            proj_in_dim += 1
        
        self.input_proj = nn.Linear(proj_in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, cond_data):
        """
        Args:
            x: (B, V, F_in) input node features (positions)
            cond_data: PyG Data object with:
                - x: (V, cond_features) node conditioning (sizes, types)
                - Optional mask field
        
        Returns:
            h: (B, V, hidden_dim)
        """
        B, V, F = x.shape
        
        # Encode positions
        pos_enc = self.pos_encoder(x)  # (B, V, pos_encoding_dim)
        
        # Get conditioning features
        cond_x = cond_data.x  # (V, cond_features)
        cond_x = cond_x.unsqueeze(0).expand(B, -1, -1)  # (B, V, cond_features)
        
        # Concatenate all features
        features = [pos_enc, x, cond_x]
        
        # Add mask if available
        if self.mask_key is not None and hasattr(cond_data, self.mask_key):
            mask = getattr(cond_data, self.mask_key).float()  # (V,)
            mask = mask.view(1, V, 1).expand(B, -1, -1)  # (B, V, 1)
            features.append(mask)
        
        h = torch.cat(features, dim=-1)
        h = self.input_proj(h)  # (B, V, hidden_dim)
        
        # Add spatial encoding if enabled
        if self.encoding_proj is not None:
            spatial_input = torch.cat([x, cond_x], dim=-1)  # (B, V, F+cond)
            spatial_enc = self._get_input_encoding(spatial_input)
            h = h + self.encoding_proj(spatial_enc)
        
        h = self.norm(h)
        return h

    def _get_input_encoding(self, spatial_input):
        B, V, D = spatial_input.shape
        theta = spatial_input.unsqueeze(dim=-1) * self.input_encoding_freqs
        embedding = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        embedding = embedding.view(B, V, D * self.input_encoding_dim)
        return embedding


class TimeConditioningMLP(nn.Module):
    """Projects time embedding into conditioning vector for FiLM."""
    
    def __init__(self, t_encoding_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(t_encoding_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        self.net = nn.Sequential(*layers)

    def forward(self, t_embed):
        return self.net(t_embed)


class OutputHead(nn.Module):
    """Output projection with zero-init for training stability.
    
    Zero-init ensures the model outputs near-zero at initialization,
    which helps with training stability in diffusion models.
    """
    
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.pre_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.final_proj = nn.Linear(hidden_dim, out_dim)
        
        # Zero-init the final projection
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.pre_proj(x)
        x = self.act(x)
        x = self.final_proj(x)
        return x


class DiscreteRotationHead(nn.Module):
    """Discrete rotation prediction with Gumbel-Softmax.
    
    Predicts one of 4 rotation classes: 0°, 90°, 180°, 270°.
    Uses Gumbel-Softmax for differentiable discrete sampling during training.
    
    Args:
        hidden_dim: Input feature dimension
        num_rotations: Number of rotation classes (default: 4)
        temperature: Initial Gumbel-Softmax temperature (default: 1.0)
        hard: If True, use straight-through estimator (hard sampling with soft gradients)
    """
    
    # Rotation transformation matrices for 0°, 90°, 180°, 270° (CCW)
    ROTATION_MATRICES = torch.tensor([
        [[1, 0], [0, 1]],      # 0°
        [[0, -1], [1, 0]],     # 90° CCW
        [[-1, 0], [0, -1]],    # 180°
        [[0, 1], [-1, 0]],     # 270° CCW
    ], dtype=torch.float32)
    
    def __init__(self, hidden_dim: int, num_rotations: int = 4, temperature: float = 1.0, hard: bool = True):
        super().__init__()
        self.num_rotations = num_rotations
        self.temperature = temperature
        self.hard = hard
        
        # Projection layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.pre_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.logits_proj = nn.Linear(hidden_dim, num_rotations)
        
        # Zero-init for stability (start with uniform predictions)
        nn.init.zeros_(self.logits_proj.weight)
        nn.init.zeros_(self.logits_proj.bias)
    
    def forward(self, h: torch.Tensor, temperature: float = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            h: (B, V, hidden_dim) node features from GNN backbone
            temperature: Override temperature (optional)
            
        Returns:
            rotation_onehot: (B, V, num_rotations) one-hot or soft rotation probabilities
            rotation_logits: (B, V, num_rotations) raw logits for loss computation
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Compute logits
        h = self.norm(h)
        h = self.pre_proj(h)
        h = self.act(h)
        logits = self.logits_proj(h)  # (B, V, num_rotations)
        
        if self.training:
            # Gumbel-Softmax: differentiable discrete sampling
            rotation_onehot = F.gumbel_softmax(logits, tau=temp, hard=self.hard)
        else:
            # Hard argmax during inference
            indices = logits.argmax(dim=-1)  # (B, V)
            rotation_onehot = F.one_hot(indices, self.num_rotations).float()
        
        return rotation_onehot, logits
    
    def get_rotation_matrices(self, rotation_onehot: torch.Tensor) -> torch.Tensor:
        """
        Get weighted rotation matrices from one-hot/soft rotation.
        
        Args:
            rotation_onehot: (B, V, num_rotations) rotation probabilities
            
        Returns:
            R: (B, V, 2, 2) rotation matrices
        """
        device = rotation_onehot.device
        R_all = self.ROTATION_MATRICES.to(device)  # (4, 2, 2)
        # Weighted sum: (B, V, 4) @ (4, 2, 2) -> (B, V, 2, 2)
        R = torch.einsum('bvk,kij->bvij', rotation_onehot, R_all)
        return R
    
    @staticmethod
    def compute_effective_size(sizes: torch.Tensor, rotation_onehot: torch.Tensor) -> torch.Tensor:
        """
        Compute effective sizes after rotation.
        
        90° and 270° rotations swap width and height.
        
        Args:
            sizes: (B, V, 2) or (V, 2) - original (width, height)
            rotation_onehot: (B, V, 4) - one-hot [0°, 90°, 180°, 270°]
            
        Returns:
            effective_sizes: (B, V, 2) - (w_eff, h_eff)
        """
        if sizes.dim() == 2:
            sizes = sizes.unsqueeze(0)  # (1, V, 2)
        
        # Rotations 90° (index 1) and 270° (index 3) swap w/h
        swap_prob = rotation_onehot[..., 1] + rotation_onehot[..., 3]  # (B, V)
        swap_prob = swap_prob.unsqueeze(-1)  # (B, V, 1)
        
        w = sizes[..., 0:1]  # (B, V, 1)
        h = sizes[..., 1:2]  # (B, V, 1)
        
        w_eff = w * (1 - swap_prob) + h * swap_prob
        h_eff = h * (1 - swap_prob) + w * swap_prob
        
        return torch.cat([w_eff, h_eff], dim=-1)


class VectorGNN(nn.Module):
    """Complete VectorGNN backbone for DiffPlace.
    
    Architecture:
    1. Input Projection: Encode positions + macro features
    2. Time Conditioning: MLP to create FiLM conditioning
    3. Encoder Body: Stack of VectorGNNBlocks with FiLM conditioning
    4. Output Head: Zero-init projection for noise prediction
    
    Features:
    - Vector-wise message passing with relative position encoding
    - FiLM conditioning at every layer
    - Zero-init output for training stability
    - Compatible with discrete timesteps (DDPM)
    - Supports 2D positions (x, y) or 3D with rotation (x, y, r)
    
    Note on Rotation:
    - Message passing always uses only 2D positions (x, y) for relative encoding
    - Rotation channel (if present) is handled separately at input/output
    - Output dimension matches out_node_features (2 or 3)
    """
    
    def __init__(
        self,
        in_node_features=2,       # input dim (x, y positions)
        out_node_features=2,      # output dim (predicted noise)
        hidden_size=128,
        t_encoding_dim=32,        # time embedding dimension
        cond_node_features=2,     # conditioning features (w, h)
        edge_features=4,
        num_blocks=4,
        layers_per_block=2,
        num_heads=4,
        pos_encoding_dim=32,
        input_encoding_dim=32,
        dropout=0.0,
        mask_key=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_size = hidden_size
        self.mask_key = mask_key
        self.device = device

        # Input projection
        # Always use 2D positions (x, y) for message passing
        # Even if out_node_features=3 (for rotation), input projection uses 2D
        input_proj_features = min(in_node_features, 2)  # Use 2D for message passing
        self.input_proj = InputProjection(
            in_node_features=input_proj_features,  # Always 2 for VectorGNN
            cond_node_features=cond_node_features,
            hidden_dim=hidden_size,
            pos_encoding_dim=pos_encoding_dim,
            input_encoding_dim=input_encoding_dim,
            mask_key=mask_key,
        )

        # Time conditioning MLP
        self.time_cond_mlp = TimeConditioningMLP(
            t_encoding_dim=t_encoding_dim,
            hidden_dim=hidden_size,
            num_layers=2,
        )

        # GNN blocks with FiLM conditioning
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                VectorGNNBlock(
                    hidden_dim=hidden_size,
                    t_cond_dim=hidden_size,  # t_cond comes from time_cond_mlp
                    pos_encoding_dim=pos_encoding_dim,
                    num_heads=num_heads,
                    num_layers=layers_per_block,
                    dropout=dropout,
                    use_edge_attr=True,
                    edge_dim=edge_features,
                )
            )

        # Output head with zero-init
        self.output_head = OutputHead(hidden_size, out_node_features)

        # Skip connection projection
        # We always use 2D positions for input (even if input has 3 channels)
        # So skip_proj maps from 2D to out_node_features (which can be 2 or 3)
        input_dim_for_skip = min(in_node_features, 2)  # Always use 2D for skip
        if input_dim_for_skip != out_node_features:
            self.skip_proj = nn.Linear(input_dim_for_skip, out_node_features)
        else:
            self.skip_proj = None

    def forward(self, x, cond, t_embed):
        """
        Forward pass compatible with DiffPlaceModel.
        
        Args:
            x: (B, V, F_in) input node features (noisy positions)
                - F_in can be 2 (x, y) or 3 (x, y, rotation)
            cond: PyG Data object with:
                - x: (V, cond_features) node conditioning (sizes)
                - edge_index: (2, E) edges
                - edge_attr: (E, edge_features) edge attributes
            t_embed: (B, t_encoding_dim) time embeddings
        
        Returns:
            out: (B, V, F_out) predicted noise
                - F_out matches out_node_features (2 or 3)
        """
        B, V, F = x.shape
        
        # Store original input for skip connection
        x_skip = x
        
        # For message passing, only use 2D positions (x, y)
        # If input has rotation channel (3D), it's not used in message passing
        # InputProjection always expects 2D positions for relative encoding
        pos = x[:, :, :2]  # (B, V, 2) - only use first 2 channels for relative position
        
        # Project inputs (InputProjection always uses 2D positions)
        # Note: self.input_proj was initialized with in_node_features=2 for VectorGNN
        # even if out_node_features=3 for rotation support
        h = self.input_proj(pos, cond)  # (B, V, hidden_size)
        
        # Get time conditioning for FiLM
        t_cond = self.time_cond_mlp(t_embed)  # (B, hidden_size)
        
        # Get graph structure from conditioning
        edge_index = cond.edge_index  # (2, E)
        edge_attr = cond.edge_attr if hasattr(cond, 'edge_attr') else None
        
        # Apply GNN blocks (using 2D positions for relative encoding)
        for block in self.blocks:
            h = block(h, pos, edge_index, t_cond, edge_attr)
        
        # Output projection (predicts noise for all output channels)
        out = self.output_head(h)  # (B, V, F_out) where F_out = out_node_features
        
        # Skip connection
        # Always use 2D positions (x, y) for skip connection
        # This ensures consistency with message passing
        x_skip_2d = x_skip[:, :, :2]  # (B, V, 2) - always use first 2 channels
        
        if self.skip_proj is not None:
            # Project 2D input to match output dimensions (2 -> 3 for rotation, or 2 -> 2)
            x_skip_proj = self.skip_proj(x_skip_2d)  # (B, V, out_node_features)
            out = out + x_skip_proj
        else:
            # Direct addition when dimensions match (both are 2D)
            out = out + x_skip_2d
        
        return out


class VectorGNNLarge(VectorGNN):
    """Larger variant of VectorGNN for complex designs."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_size', 256)
        kwargs.setdefault('num_blocks', 5)
        kwargs.setdefault('layers_per_block', 3)
        kwargs.setdefault('num_heads', 8)
        kwargs.setdefault('pos_encoding_dim', 64)
        kwargs.setdefault('input_encoding_dim', 64)
        super().__init__(**kwargs)


class VectorGNNSmall(VectorGNN):
    """Smaller variant of VectorGNN for faster iteration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_size', 64)
        kwargs.setdefault('num_blocks', 2)
        kwargs.setdefault('layers_per_block', 2)
        kwargs.setdefault('num_heads', 4)
        kwargs.setdefault('pos_encoding_dim', 16)
        kwargs.setdefault('input_encoding_dim', 16)
        super().__init__(**kwargs)


class VectorGNNV2(nn.Module):
    """VectorGNN V2 with bifurcated outputs for discrete rotation.
    
    Key differences from VectorGNN:
    1. Separate output heads: position (continuous) + rotation (discrete 4-class)
    2. Position head predicts noise epsilon (for MSE loss)
    3. Rotation head uses Gumbel-Softmax (for CrossEntropy loss)
    4. Supports temperature annealing for rotation prediction
    
    Forward returns a tuple:
        - pos_pred: (B, V, 2) position noise prediction
        - rot_onehot: (B, V, 4) discrete rotation one-hot
        - rot_logits: (B, V, 4) rotation logits for CE loss
    """
    
    def __init__(
        self,
        in_node_features: int = 2,        # input dim (x, y positions only)
        hidden_size: int = 128,
        t_encoding_dim: int = 32,
        cond_node_features: int = 2,      # (width, height)
        edge_features: int = 4,
        num_blocks: int = 4,
        layers_per_block: int = 2,
        num_heads: int = 4,
        pos_encoding_dim: int = 32,
        input_encoding_dim: int = 32,
        dropout: float = 0.0,
        mask_key: str = None,
        num_rotations: int = 4,
        rotation_temperature: float = 1.0,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.in_node_features = in_node_features
        self.hidden_size = hidden_size
        self.mask_key = mask_key
        self.device = device
        self.num_rotations = num_rotations
        
        # Input projection (always 2D positions)
        self.input_proj = InputProjection(
            in_node_features=2,  # Always 2D for VectorGNN message passing
            cond_node_features=cond_node_features,
            hidden_dim=hidden_size,
            pos_encoding_dim=pos_encoding_dim,
            input_encoding_dim=input_encoding_dim,
            mask_key=mask_key,
        )
        
        # Time conditioning MLP
        self.time_cond_mlp = TimeConditioningMLP(
            t_encoding_dim=t_encoding_dim,
            hidden_dim=hidden_size,
            num_layers=2,
        )
        
        # GNN blocks (shared backbone)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                VectorGNNBlock(
                    hidden_dim=hidden_size,
                    t_cond_dim=hidden_size,
                    pos_encoding_dim=pos_encoding_dim,
                    num_heads=num_heads,
                    num_layers=layers_per_block,
                    dropout=dropout,
                    use_edge_attr=True,
                    edge_dim=edge_features,
                )
            )
        
        # === Bifurcated Output Heads ===
        # Position head: predicts noise (continuous, MSE loss)
        self.position_head = OutputHead(hidden_size, out_dim=2)
        
        # Rotation head: predicts discrete class (Gumbel-Softmax, CE loss)
        self.rotation_head = DiscreteRotationHead(
            hidden_dim=hidden_size,
            num_rotations=num_rotations,
            temperature=rotation_temperature,
            hard=True,
        )
        
        # Skip connection for positions
        self.skip_proj = None  # in_node_features == 2 == out_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        cond, 
        t_embed: torch.Tensor,
        rotation_temperature: float = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with bifurcated outputs.
        
        Args:
            x: (B, V, 2) input noisy positions (x, y only, no rotation)
            cond: PyG Data object with node/edge features
            t_embed: (B, t_encoding_dim) time embeddings
            rotation_temperature: Optional temperature override for Gumbel-Softmax
            
        Returns:
            pos_pred: (B, V, 2) predicted position noise
            rot_onehot: (B, V, num_rotations) discrete rotation one-hot
            rot_logits: (B, V, num_rotations) rotation logits for CrossEntropy
        """
        B, V, _ = x.shape
        
        # Store for skip connection
        x_skip = x[:, :, :2]  # Always 2D
        pos = x_skip  # Use 2D positions for message passing
        
        # Project inputs
        h = self.input_proj(pos, cond)  # (B, V, hidden_size)
        
        # Time conditioning
        t_cond = self.time_cond_mlp(t_embed)  # (B, hidden_size)
        
        # Get graph structure
        edge_index = cond.edge_index
        edge_attr = cond.edge_attr if hasattr(cond, 'edge_attr') else None
        
        # Apply GNN blocks
        for block in self.blocks:
            h = block(h, pos, edge_index, t_cond, edge_attr)
        
        # === Bifurcated Outputs ===
        # Position prediction (with skip connection)
        pos_pred = self.position_head(h)  # (B, V, 2)
        pos_pred = pos_pred + x_skip  # Skip connection
        
        # Rotation prediction (discrete)
        rot_onehot, rot_logits = self.rotation_head(h, temperature=rotation_temperature)
        
        return pos_pred, rot_onehot, rot_logits
    
    def get_effective_sizes(self, cond, rotation_onehot: torch.Tensor) -> torch.Tensor:
        """
        Compute effective macro sizes after rotation.
        
        Args:
            cond: PyG Data with cond.x = (V, 2) original sizes
            rotation_onehot: (B, V, 4) rotation probabilities
            
        Returns:
            effective_sizes: (B, V, 2)
        """
        return DiscreteRotationHead.compute_effective_size(cond.x, rotation_onehot)


class VectorGNNV2Large(VectorGNNV2):
    """Larger variant of VectorGNNV2."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_size', 256)
        kwargs.setdefault('num_blocks', 5)
        kwargs.setdefault('layers_per_block', 3)
        kwargs.setdefault('num_heads', 8)
        kwargs.setdefault('pos_encoding_dim', 64)
        kwargs.setdefault('input_encoding_dim', 64)
        super().__init__(**kwargs)


# ===================== GLOBAL CONTEXT MODULE =====================

class GlobalContextModule(nn.Module):
    """
    Virtual Global Supernode for O(V) global context injection.
    
    Mechanism:
    1. READ (Aggregate): Mean pool all node features -> global token
    2. PROCESS: Transform global token via MLP
    3. WRITE (Broadcast): Add global context back to all nodes
    
    Complexity: O(V) - linear in number of nodes
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable global token (bias for aggregation)
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.global_token, std=0.02)
        
        # Process MLP: global_dim -> global_dim
        self.process_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Gate for soft blending (learn how much global to mix in)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        # LayerNorm for output
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Zero-init for stable training (start as identity)
        nn.init.zeros_(self.process_mlp[-2].weight)
        nn.init.zeros_(self.process_mlp[-2].bias)
    
    def forward(
        self, 
        h: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Inject global context into node features.
        
        Args:
            h: (B, V, D) node features
            mask: (B, V) optional boolean mask (True = ignore in pooling)
            
        Returns:
            h_out: (B, V, D) node features with global context
        """
        B, V, D = h.shape
        
        # === READ: Global aggregation ===
        if mask is not None:
            # Masked mean pooling
            mask_float = (~mask).float().unsqueeze(-1)  # (B, V, 1)
            h_masked = h * mask_float
            global_agg = h_masked.sum(dim=1, keepdim=True) / (mask_float.sum(dim=1, keepdim=True) + 1e-8)
        else:
            global_agg = h.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # Add learnable global token
        global_agg = global_agg + self.global_token  # (B, 1, D)
        
        # === PROCESS: Transform global context ===
        global_processed = self.process_mlp(global_agg)  # (B, 1, D)
        
        # === WRITE: Broadcast back to all nodes ===
        global_broadcast = global_processed.expand(B, V, D)  # (B, V, D)
        
        # Gated addition
        gate_input = torch.cat([h, global_broadcast], dim=-1)  # (B, V, 2D)
        gate = self.gate(gate_input)  # (B, V, D)
        
        h_out = h + gate * global_broadcast
        h_out = self.norm(h_out)
        
        return h_out


class VectorGNNV2Global(nn.Module):
    """
    VectorGNN V2 with Global Context injection.
    
    Combines:
    - VectorGNNV2 bifurcated outputs (position + rotation)
    - GlobalContextModule after each block for O(V) global awareness
    
    This allows nodes to "see" the entire graph after just 1 layer.
    """
    
    def __init__(
        self,
        in_node_features: int = 2,
        hidden_size: int = 128,
        t_encoding_dim: int = 32,
        cond_node_features: int = 2,
        edge_features: int = 4,
        num_blocks: int = 4,
        layers_per_block: int = 2,
        num_heads: int = 4,
        pos_encoding_dim: int = 32,
        input_encoding_dim: int = 32,
        dropout: float = 0.0,
        mask_key: str = None,
        num_rotations: int = 4,
        rotation_temperature: float = 1.0,
        global_context_every: int = 1,  # Apply global context every N blocks
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.in_node_features = in_node_features
        self.hidden_size = hidden_size
        self.mask_key = mask_key
        self.device = device
        self.num_rotations = num_rotations
        self.global_context_every = global_context_every
        
        # Input projection
        self.input_proj = InputProjection(
            in_node_features=2,
            cond_node_features=cond_node_features,
            hidden_dim=hidden_size,
            pos_encoding_dim=pos_encoding_dim,
            input_encoding_dim=input_encoding_dim,
            mask_key=mask_key,
        )
        
        # Time conditioning
        self.time_cond_mlp = TimeConditioningMLP(
            t_encoding_dim=t_encoding_dim,
            hidden_dim=hidden_size,
            num_layers=2,
        )
        
        # GNN blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                VectorGNNBlock(
                    hidden_dim=hidden_size,
                    t_cond_dim=hidden_size,
                    pos_encoding_dim=pos_encoding_dim,
                    num_heads=num_heads,
                    num_layers=layers_per_block,
                    dropout=dropout,
                    use_edge_attr=True,
                    edge_dim=edge_features,
                )
            )
        
        # Global context modules (one per block where we apply it)
        num_global = (num_blocks + global_context_every - 1) // global_context_every
        self.global_contexts = nn.ModuleList([
            GlobalContextModule(hidden_size, dropout=dropout)
            for _ in range(num_global)
        ])
        
        # Output heads
        self.position_head = OutputHead(hidden_size, out_dim=2)
        self.rotation_head = DiscreteRotationHead(
            hidden_dim=hidden_size,
            num_rotations=num_rotations,
            temperature=rotation_temperature,
            hard=True,
        )
        
        self.skip_proj = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        cond, 
        t_embed: torch.Tensor,
        rotation_temperature: float = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with global context injection.
        """
        B, V, _ = x.shape
        
        x_skip = x[:, :, :2]
        pos = x_skip
        
        # Get mask for global pooling
        mask = None
        if self.mask_key and hasattr(cond, self.mask_key):
            mask = getattr(cond, self.mask_key)  # (V,)
            mask = mask.unsqueeze(0).expand(B, -1)  # (B, V)
        
        # Project inputs
        h = self.input_proj(pos, cond)
        t_cond = self.time_cond_mlp(t_embed)
        
        edge_index = cond.edge_index
        edge_attr = cond.edge_attr if hasattr(cond, 'edge_attr') else None
        
        # Apply blocks with global context
        global_idx = 0
        for i, block in enumerate(self.blocks):
            h = block(h, pos, edge_index, t_cond, edge_attr)
            
            # Apply global context every N blocks
            if (i + 1) % self.global_context_every == 0:
                h = self.global_contexts[global_idx](h, mask)
                global_idx += 1
        
        # Output heads
        pos_pred = self.position_head(h) + x_skip
        rot_onehot, rot_logits = self.rotation_head(h, temperature=rotation_temperature)
        
        return pos_pred, rot_onehot, rot_logits
    
    def get_effective_sizes(self, cond, rotation_onehot: torch.Tensor) -> torch.Tensor:
        return DiscreteRotationHead.compute_effective_size(cond.x, rotation_onehot)


class VectorGNNV2GlobalLarge(VectorGNNV2Global):
    """Larger variant with global context."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_size', 256)
        kwargs.setdefault('num_blocks', 5)
        kwargs.setdefault('layers_per_block', 3)
        kwargs.setdefault('num_heads', 8)
        kwargs.setdefault('pos_encoding_dim', 64)
        kwargs.setdefault('input_encoding_dim', 64)
        super().__init__(**kwargs)
