import torch
import torch.nn.functional as F
import torch_geometric.nn as tgn

BLOCK_SIZE = 16384 # 8192 # 16384


def compute_effective_size(sizes, rot_logits=None):
    """
    Compute effective size considering rotation.
    
    Args:
        sizes: (B, V, 2) or (V, 2) - original sizes (w, h)
        rot_logits: (B, V) or (B, V, 1) or None - rotation logits
    
    Returns:
        effective_sizes: same shape as sizes - (w_eff, h_eff)
        
    When rot_logits is provided:
        prob = sigmoid(rot_logits)
        w_eff = w * (1 - prob) + h * prob  # interpolate between w and h
        h_eff = h * (1 - prob) + w * prob  # interpolate between h and w
        
    This allows gradients to flow through rot_logits for optimization.
    When prob=0 (no rotation): (w_eff, h_eff) = (w, h)
    When prob=1 (90° rotation): (w_eff, h_eff) = (h, w)
    """
    if rot_logits is None:
        return sizes
    
    # Ensure rot_logits has correct shape
    if rot_logits.dim() == 2:
        rot_logits = rot_logits.unsqueeze(-1)  # (B, V) -> (B, V, 1)
    
    prob = torch.sigmoid(rot_logits)  # (B, V, 1)
    
    # Ensure sizes has batch dimension
    if sizes.dim() == 2:
        sizes = sizes.unsqueeze(0)  # (V, 2) -> (1, V, 2)
    
    w = sizes[..., 0:1]  # (B, V, 1)
    h = sizes[..., 1:2]  # (B, V, 1)
    
    w_eff = w * (1 - prob) + h * prob
    h_eff = h * (1 - prob) + w * prob
    
    return torch.cat([w_eff, h_eff], dim=-1)  # (B, V, 2)


# ============== V2 Functions for Discrete Rotation ==============

# Rotation transformation matrices for 0°, 90°, 180°, 270° (CCW)
ROTATION_MATRICES_4 = torch.tensor([
    [[1, 0], [0, 1]],      # 0°: identity
    [[0, -1], [1, 0]],     # 90° CCW
    [[-1, 0], [0, -1]],    # 180°
    [[0, 1], [-1, 0]],     # 270° CCW
], dtype=torch.float32)


def compute_effective_size_v2(sizes: torch.Tensor, rotation_onehot: torch.Tensor = None) -> torch.Tensor:
    """
    Compute effective size considering discrete 4-class rotation.
    
    This is the V2 version that works with Gumbel-Softmax one-hot rotation.
    
    Args:
        sizes: (B, V, 2) or (V, 2) - original sizes (w, h)
        rotation_onehot: (B, V, 4) - one-hot for [0°, 90°, 180°, 270°]
        
    Returns:
        effective_sizes: (B, V, 2) - (w_eff, h_eff)
        
    When rotation is 90° or 270°, width and height are swapped.
    Uses soft interpolation for gradient flow during training.
    """
    if rotation_onehot is None:
        if sizes.dim() == 2:
            return sizes.unsqueeze(0)
        return sizes
    
    # Ensure sizes has batch dimension
    if sizes.dim() == 2:
        sizes = sizes.unsqueeze(0)  # (V, 2) -> (1, V, 2)
    
    # Expand sizes to match batch if needed
    B = rotation_onehot.shape[0]
    if sizes.shape[0] == 1 and B > 1:
        sizes = sizes.expand(B, -1, -1)
    
    # Rotations 90° (index 1) and 270° (index 3) swap w/h
    swap_prob = rotation_onehot[..., 1] + rotation_onehot[..., 3]  # (B, V)
    swap_prob = swap_prob.unsqueeze(-1)  # (B, V, 1)
    
    w = sizes[..., 0:1]  # (B, V, 1)
    h = sizes[..., 1:2]  # (B, V, 1)
    
    w_eff = w * (1 - swap_prob) + h * swap_prob
    h_eff = h * (1 - swap_prob) + w * swap_prob
    
    return torch.cat([w_eff, h_eff], dim=-1)  # (B, V, 2)


def compute_rotated_pin_offsets_v2(
    pin_offsets: torch.Tensor, 
    rotation_onehot: torch.Tensor, 
    pin_map: torch.Tensor
) -> torch.Tensor:
    """
    Compute rotated pin offsets using discrete 4-class rotation.
    
    Args:
        pin_offsets: (P, 2) - original pin offsets from macro centers
        rotation_onehot: (B, V, 4) - one-hot rotation for each macro
        pin_map: (P,) - mapping from pin index to macro index
        
    Returns:
        rotated_offsets: (B, P, 2) - rotated pin offsets
    """
    device = pin_offsets.device
    B = rotation_onehot.shape[0]
    P = pin_offsets.shape[0]
    
    # Get rotation for each pin's parent macro
    macro_rot = rotation_onehot[:, pin_map, :]  # (B, P, 4)
    
    # Get rotation matrices
    R_all = ROTATION_MATRICES_4.to(device)  # (4, 2, 2)
    
    # Weighted rotation matrix: (B, P, 4) @ (4, 2, 2) -> (B, P, 2, 2)
    R = torch.einsum('bpk,kij->bpij', macro_rot, R_all)
    
    # Apply rotation: (B, P, 2, 2) @ (P, 2) -> (B, P, 2)
    rotated = torch.einsum('bpij,pj->bpi', R, pin_offsets)
    
    return rotated


def legality_guidance_potential_v2(
    positions: torch.Tensor, 
    cond, 
    rotation_onehot: torch.Tensor = None,
    softmax_factor: float = 10.0, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    V2 Legality guidance potential for discrete rotation.
    
    Args:
        positions: (B, V, 2) - macro positions (x, y)
        cond: PyG Data object with cond.x = (V, 2) sizes
        rotation_onehot: (B, V, 4) - one-hot rotation probabilities
        softmax_factor: Temperature for soft-min
        mask: (B, V, 1) or None - masked nodes are ignored
        
    Returns:
        h_total: (B,) - scalar potential per batch
    """
    B, V, D = positions.shape
    
    # Compute effective sizes considering rotation
    raw_sizes = cond.x  # (V, 2)
    sizes = compute_effective_size_v2(raw_sizes, rotation_onehot)  # (B, V, 2)
    
    # Pairwise overlap computation
    x_1 = positions.view(B, V, 1, D)
    x_2 = positions.view(B, 1, V, D).detach()
    size_1 = sizes.view(B, V, 1, D)
    size_2 = sizes.view(B, 1, V, D)
    
    delta = torch.abs(x_1 - x_2) - ((size_1 + size_2) / 2)  # (B, V, V, D)
    l = torch.sum(F.softmax(delta * softmax_factor, dim=-1) * delta, dim=-1, keepdim=True)
    h = (F.relu(-l) ** 2) / 4
    
    # Boundary term
    h_bound = (F.relu(torch.abs(positions) + sizes / 2 - 1) ** 2) / 2  # (B, V, D)
    
    # Mask self-collisions and optionally masked nodes
    mask_square = (1 - torch.eye(V, dtype=h.dtype, device=h.device)).view(1, V, V, 1)
    if mask is not None:
        inv_mask = ~mask
        mask_square = mask_square * inv_mask.view(B, 1, V, 1) * inv_mask.view(B, V, 1, 1)
        h_bound = inv_mask.float() * h_bound
    
    # Weight by mass
    mass_1 = torch.exp(torch.mean(torch.log(sizes + 1e-8), dim=-1, keepdim=True)).unsqueeze(-1)
    mass_2 = mass_1.view(B, 1, V, 1)
    h = h * mask_square * (mass_2 / (mass_1 + mass_2 + 1e-8))
    
    # Sum over all dimensions
    h_dims = list(range(1, len(h.shape)))
    h_bound_dims = list(range(1, len(h_bound.shape)))
    h_total = h.sum(dim=h_dims) + h_bound.sum(dim=h_bound_dims)
    
    return h_total


def extract_positions_and_rotation(x_hat):
    """
    Extract 2D positions and optional rotation from input.
    
    Args:
        x_hat: (B, V, 2) or (B, V, 3) - positions with optional rotation
        
    Returns:
        positions: (B, V, 2) - (x, y) coordinates
        rot_logits: (B, V, 1) or None - rotation logits if present
    """
    if x_hat.shape[-1] == 2:
        return x_hat, None
    elif x_hat.shape[-1] == 3:
        positions = x_hat[..., :2]  # (B, V, 2)
        rot_logits = x_hat[..., 2:3]  # (B, V, 1)
        return positions, rot_logits
    else:
        raise ValueError(f"Expected x_hat with 2 or 3 channels, got {x_hat.shape[-1]}")


def legality_guidance_potential(x_hat, cond, softmax_factor=10.0, mask=None):
    """
    Differentiable function for computing legality guidance potential
    Inputs:
    - x_hat (B, V, 2) or (B, V, 3) - positions with optional rotation logit
    - cond is pytorch data object
    
    If x_hat has 3 channels, the third channel is used as rotation logit
    to compute effective sizes via soft interpolation.
    """
    # Extract positions and optional rotation
    positions, rot_logits = extract_positions_and_rotation(x_hat)
    B, V, D = positions.shape
    
    # Compute effective sizes considering rotation
    raw_sizes = cond.x.expand(B, *cond.x.shape)  # (B, V, 2)
    sizes = compute_effective_size(raw_sizes, rot_logits)  # (B, V, 2)

    # compute energy h. we use convention that higher h = less favorable ie. force = -grad(h)
    x_1 = positions.view(B, V, 1, D)
    x_2 = positions.view(B, 1, V, D).detach()
    size_1 = sizes.view(B, V, 1, D)
    size_2 = sizes.view(B, 1, V, D)
    delta = torch.abs(x_1 - x_2) - ((size_1 + size_2)/2) # (B, V1, V2, D)
    # softmax and max both work
    l = torch.sum(F.softmax(delta * softmax_factor, dim=-1) * delta, dim=-1, keepdim=True)
    h = (F.relu(-l)**2) / 4

    # calculate boundary term (using positions and effective sizes)
    h_bound = (F.relu(torch.abs(positions) + sizes/2 - 1) ** 2)/2 # (B, V, D)

    # mask out objects where mask=True and self-collisions
    mask_square = (1-torch.eye(V, dtype=h.dtype, device=h.device)).view(1, V, V, 1) # ignore self-collision
    if mask is not None:
        inv_mask = ~mask
        mask_square = mask_square * inv_mask.view(1, 1, V, 1) * inv_mask.view(1, V, 1, 1)
        h_bound = inv_mask * h_bound
    
    # weight forces by size of instances
    mass_1 = torch.exp(torch.mean(torch.log(sizes), dim=-1, keepdim=True)).unsqueeze(dim=-1) # (B, V1, 1, 1)
    mass_2 = mass_1.view(B, 1, V, 1) # (B, 1, V2, 1)
    h = h * mask_square * ((mass_2)/(mass_1 + mass_2)) # (B, V1, V2, D)
    h_bound = h_bound

    # compute forces
    h_dims = list(range(1, len(h.shape)))
    h_bound_dims = list(range(1, len(h_bound.shape)))
    h_total = h.sum(dim = h_dims) + h_bound.sum(dim = h_bound_dims)
    return h_total

def legality_guidance_potential_tiled(x_hat, cond, softmax_factor=10.0, mask=None, block_size=BLOCK_SIZE):
    """
    Differentiable function for computing legality guidance potential
    This is a tiled, memory-constrained version of the above
    Inputs:
    - x_hat (B, V, 2) or (B, V, 3) - positions with optional rotation logit
    - cond is pytorch data object

    NOTE this function performs backward passes wrt x_hat to save memory
    Output:
    - Detached legality potential
    """
    # Extract positions and optional rotation
    positions, rot_logits = extract_positions_and_rotation(x_hat)
    B, V, D = positions.shape
    
    # Compute effective sizes considering rotation
    raw_sizes = cond.x  # (V, 2)
    if rot_logits is not None:
        sizes = compute_effective_size(raw_sizes.unsqueeze(0).expand(B, -1, -1), rot_logits)
        sizes = sizes[0]  # Use first batch for tile computation (sizes are per-node, not per-batch)
    else:
        sizes = raw_sizes

    h_bound = legality_potential_boundary(x_hat, cond, mask=mask)
    h = 0
    for start_i in range(0, V, block_size):
        end_i = min(start_i + block_size, V)
        for start_j in range(0, V, block_size):
            end_j = min(start_j + block_size, V)
            h_current = legality_potential_tile(
                positions[:, start_i:end_i, :], 
                positions[:, start_j:end_j, :], 
                sizes[start_i:end_i, :],
                sizes[start_j:end_j, :],
                start_i == start_j,
                mask_1 = None if mask is None else mask[:, start_i:end_i, :],
                mask_2 = None if mask is None else mask[:, start_j:end_j, :],
                softmax_factor = softmax_factor,
                )
            # backward pass to save memory
            h_current.backward()
            h = h + h_current.detach()
    h_bound.backward()
    h_total = h + h_bound.detach()
    return h_total

def legality_potential_tile(x_hat_1, x_hat_2, size_1, size_2, is_diagonal, mask_1 = None, mask_2 = None, softmax_factor=10.0):
    """
    Differentiable function for computing legality guidance potential
    Inputs:
    - x_hat (B, V, 2)
    - size (V, 2)
    - masks are (1, V, 2) or (B, V, 2)
    - is_diagonal specifies if self-collisions should be masked out
    TODO make it so that we don't have to duplicate computations for the lower triangle
    NOTE this is possible by multiplying m1*m2/(m1+m2) and scaling gradients w.r.t m1 before optimizer.step
    NOTE if this is to be done, we also have to be careful of the diagonal and to not detach x_2
    """
    B, V_1, D = x_hat_1.shape
    B_2, V_2, D_2 = x_hat_2.shape
    assert (B == B_2) and (D == D_2), "input x must have same batch and feature dimensions"

    # compute energy h. we use convention that higher h = less favorable ie. force = -grad(h)
    x_1 = x_hat_1.view(B, V_1, 1, D)
    x_2 = x_hat_2.view(B, 1, V_2, D).detach()
    size_1 = size_1.expand(B, *size_1.shape).view(B, V_1, 1, D)
    size_2 = size_2.expand(B, *size_2.shape).view(B, 1, V_2, D)
    delta = torch.abs(x_1 - x_2) - ((size_1 + size_2)/2) # (B, V1, V2, D)
    # softmax and max both work
    l = torch.sum(F.softmax(delta * softmax_factor, dim=-1) * delta, dim=-1, keepdim=True)
    h = (F.relu(-l)**2) / 4

    # mask out objects where mask=True and self-collisions
    if is_diagonal:
        mask_square = (1-torch.eye(n=V_1, m=V_2, dtype=h.dtype, device=h.device)).view(1, V_1, V_2, 1) # ignore self-collision
    else:
        mask_square = 1
    if (mask_1 is not None) and (mask_2 is not None):
        inv_mask_1 = ~mask_1
        inv_mask_2 = ~mask_2
        mask_square = mask_square * inv_mask_1.view(1, V_1, 1, 1) * inv_mask_2.view(1, 1, V_2, 1)
    
    # weight forces by size of instances
    mass_1 = torch.exp(torch.mean(torch.log(size_1), dim=-1, keepdim=True)) # (B, V1, 1, 1)
    mass_2 = torch.exp(torch.mean(torch.log(size_2), dim=-1, keepdim=True)) # (B, 1, V2, 1)
    h = h * mask_square * ((mass_2)/(mass_1 + mass_2)) # (B, V1, V2, D)

    # compute forces
    h_dims = list(range(1, len(h.shape)))
    h_tile = h.sum(dim = h_dims)
    return h_tile

def legality_potential_boundary(x_hat, cond, mask=None):
    """
    Differentiable function for computing boundary-enforcement term of legality guidance potential
    Inputs:
    - x_hat (B, V, 2) or (B, V, 3) - positions with optional rotation logit
    - cond is pytorch data object
    """
    # Extract positions and optional rotation
    positions, rot_logits = extract_positions_and_rotation(x_hat)
    B, V, D = positions.shape
    
    # Compute effective sizes considering rotation
    raw_sizes = cond.x.view(1, V, 2)  # (1, V, 2)
    sizes = compute_effective_size(raw_sizes.expand(B, -1, -1), rot_logits)  # (B, V, 2)

    # calculate boundary term
    h_bound = (F.relu(torch.abs(positions) + sizes/2 - 1) ** 2)/2 # (B, V, D)

    # mask out objects where mask=True and self-collisions
    if mask is not None:
        inv_mask = ~mask
        h_bound = inv_mask * h_bound

    # compute forces
    h_bound_dims = list(range(1, len(h_bound.shape)))
    return h_bound.sum(dim = h_bound_dims)

def compute_rotated_pin_offsets(pin_offsets, rot_logits, pin_map):
    """
    Compute rotated pin offsets based on rotation logits.
    
    Args:
        pin_offsets: (P, 2) - original pin offsets from macro centers
        rot_logits: (B, V, 1) or None - rotation logits for each macro
        pin_map: (P,) - mapping from pin to macro index
        
    Returns:
        rotated_offsets: (B, P, 2) - rotated pin offsets
        
    When macro rotates 90° CCW: (dx, dy) -> (-dy, dx)
    Soft rotation interpolates between original and rotated offsets.
    """
    if rot_logits is None:
        return pin_offsets.unsqueeze(0)  # (1, P, 2)
    
    B = rot_logits.shape[0]
    P = pin_offsets.shape[0]
    
    # Get rotation probability for each pin (via pin_map)
    macro_rot_logits = rot_logits[:, pin_map, :]  # (B, P, 1)
    prob = torch.sigmoid(macro_rot_logits)  # (B, P, 1)
    
    # Original offsets
    dx = pin_offsets[:, 0:1]  # (P, 1)
    dy = pin_offsets[:, 1:2]  # (P, 1)
    
    # Rotated offsets (90° CCW): (dx, dy) -> (-dy, dx)
    dx_rot = -dy
    dy_rot = dx
    
    # Soft interpolation
    dx_eff = dx * (1 - prob) + dx_rot * prob  # (B, P, 1)
    dy_eff = dy * (1 - prob) + dy_rot * prob  # (B, P, 1)
    
    return torch.cat([dx_eff, dy_eff], dim=-1)  # (B, P, 2)


def hpwl_guidance_potential(x, cond, pin_map=None, pin_offsets=None, pin_edge_index=None, hpwl_net=None):
    """
    Differentiable function for computing hpwl
    Inputs:
    - x (B, V, 2) or (B, V, 3) - positions with optional rotation logit
    - cond is pytorch data object with edge_index (2, E) and edge_attr (E, 4)
    - pin map, offsets, edge index, and hpwl_net are optional variables that should be cached per-netlist
    """
    # Extract positions and optional rotation
    positions, rot_logits = extract_positions_and_rotation(x)
    
    # compute netlist-level info if cached version not provided
    if pin_map is None or pin_offsets is None or pin_edge_index is None:
        pin_map, pin_offsets, pin_edge_index = compute_pin_map(cond)
    if hpwl_net is None:
        hpwl_net = HPWL()
    
    # Compute rotated pin offsets if rotation is present
    if rot_logits is not None:
        effective_pin_offsets = compute_rotated_pin_offsets(pin_offsets, rot_logits, pin_map)
    else:
        effective_pin_offsets = pin_offsets
    
    # compute and return hpwl
    hpwl = hpwl_net(positions, pin_map, effective_pin_offsets, pin_edge_index)
    return hpwl

def hpwl_square_guidance_potential(x, cond, pin_map=None, pin_offsets=None, pin_edge_index=None, hpwl_net=None):
    """
    Differentiable function for computing hpwl-based potential, using square of distance
    Inputs:
    - x (B, V, 2) or (B, V, 3) - positions with optional rotation logit
    - cond is pytorch data object with edge_index (2, E) and edge_attr (E, 4)
    - pin map, offsets, edge index, and hpwl_net are optional variables that should be cached per-netlist
    """
    # Extract positions and optional rotation
    positions, rot_logits = extract_positions_and_rotation(x)
    
    # compute netlist-level info if cached version not provided
    if pin_map is None or pin_offsets is None or pin_edge_index is None:
        pin_map, pin_offsets, pin_edge_index = compute_pin_map(cond)
    if hpwl_net is None:
        hpwl_net = HPWL()
    
    # Compute rotated pin offsets if rotation is present
    if rot_logits is not None:
        effective_pin_offsets = compute_rotated_pin_offsets(pin_offsets, rot_logits, pin_map)
    else:
        effective_pin_offsets = pin_offsets
    
    # compute and return hpwl
    hpwl = hpwl_net(positions, pin_map, effective_pin_offsets, pin_edge_index, net_aggr = "none") # (B, P)
    hpwl_potential = ((hpwl ** 2)/2).sum(dim=-1)
    return hpwl_potential

class HPWL(tgn.MessagePassing):
    def __init__(self):
        super().__init__(aggr="max", flow="target_to_source") 
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, pin_map, pin_offsets, pin_edge_index, net_aggr = "sum", raw_output = False):
        # x is (B, V, 2) matrix defining the placement
        # pin_map is (P) tensor such that pin_map[pin_idx] = object_idx
        # pin_offsets is (P, 2) or (B, P, 2) tensor - may have batch dimension for rotation
        # pin_edge_index has shape (2, E) note that this has to preprocessed
        
        # Handle pin_offsets with or without batch dimension
        macro_positions = x[..., pin_map, :]  # (B, P, 2)
        if pin_offsets.dim() == 2:
            # No batch dimension: (P, 2) -> broadcast
            global_pin_position = pin_offsets + macro_positions  # (B, P, 2)
        else:
            # Has batch dimension: (B, P, 2)
            global_pin_position = pin_offsets + macro_positions  # (B, P, 2)
        
        # Start propagating messages.
        net_maxmin = F.relu(self.propagate(pin_edge_index, x=global_pin_position)) # (B, P, 4)
        if raw_output:
            return net_maxmin
        net_hpwl = torch.sum(net_maxmin, dim=-1)
        if net_aggr == "sum":
            hpwl = torch.sum(net_hpwl, dim=-1)
        elif net_aggr == "mean":
            hpwl = torch.mean(net_hpwl, dim=-1)
        elif net_aggr == "none":
            hpwl = net_hpwl
        else:
            raise NotImplementedError
        return hpwl

    def message(self, x_i, x_j):
        # x_i and x_j has shape (B, E, 2)
        delta = x_j - x_i
        delta_combined = torch.cat((delta, -delta), dim=-1) # (B, E, 4)
        return delta_combined

class MacroHPWL(tgn.MessagePassing):
    def __init__(self):
        super().__init__(aggr="max", flow="target_to_source") 
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, pin_map, pin_offsets, pin_edge_index, is_macro, net_aggr = "sum", raw_output = False):
        # x is (B, V, 2) matrix defining the placement
        # pin_map is (P) tensor such that pin_map[pin_idx] = object_idx
        # pin_offsets is (P, 2) tensor
        # pin_edge_index has shape (2, E) note that this has to preprocessed
        global_pin_position = pin_offsets + x[..., pin_map, :] # (B, P, 2)
        pin_is_macro = is_macro[..., pin_map, None]
        # Start propagating messages.
        net_maxmin = self.propagate(pin_edge_index, x=global_pin_position, is_macro=pin_is_macro) # (B, P, 4)
        net_maxmin[net_maxmin==float('-inf')] = 0
        if raw_output:
            return net_maxmin
        net_hpwl = torch.sum(net_maxmin, dim=-1)
        if net_aggr == "sum":
            hpwl = torch.sum(net_hpwl, dim=-1)
        elif net_aggr == "mean":
            hpwl = torch.mean(net_hpwl, dim=-1)
        elif net_aggr == "none":
            hpwl = net_hpwl
        else:
            raise NotImplementedError
        return hpwl

    def message(self, x_i, x_j, is_macro_i, is_macro_j):
        # x_i and x_j has shape (B, E, 2)
        # is_macro_i and is_macro_j have shape (B, E, 1)
        data_i = torch.cat((x_i, -x_i), dim=-1) # (B, E, 4)
        data_j = torch.cat((x_j, -x_j), dim=-1) # (B, E, 4)
        
        # ignore non-macros
        data_i_masked = torch.where(is_macro_i, data_i, float('-inf'))
        data_j_masked = torch.where(is_macro_j, data_j, float('-inf'))
        data_combined_masked = torch.maximum(data_i_masked, data_j_masked) # (B, E, 4)
        return data_combined_masked

def compute_pin_map(cond):
    """
    Computes tensors needed for computing hpwl efficiently \\
    Returns:
    - Pin map: (P) tensor such that pin_map[pin_idx] = object_idx
    - Pin offsets: (P, 2) tensor with offsets for each pin
    - Pin_edge_index: edge_index, except using pin indices instead of object indices 
        (so pin_map[pin_edge_index] == edge_index_unique)
    """
    _, E = cond.edge_index.shape
    assert E % 2 == 0, "cond edge index assumed to contain forward and reverse edges"
    edge_index_unique = cond.edge_index[:, :E//2].T # (E, 2)
    edge_attr_unique = cond.edge_attr[:E//2, :]
    if "edge_pin_id" in cond:
        edge_pin_id_unique = cond.edge_pin_id[:E//2, :]

    # note: we convert to double to avoid float roundoff error for >17M edges
    if "edge_pin_id" in cond:
        sources = torch.cat((
            edge_index_unique[:,0:1].double(), 
            edge_attr_unique[:,0:2].double(), 
            edge_pin_id_unique[:,0:1].double(),
            ), dim=1)
        dests = torch.cat((
            edge_index_unique[:,1:2].double(), 
            edge_attr_unique[:,2:4].double(),
            edge_pin_id_unique[:,1:2].double(),
            ), dim=1)
    else:
        sources = torch.cat((
            edge_index_unique[:,0:1].double(), 
            edge_attr_unique[:,0:2].double()
            ), dim=1)
        dests = torch.cat((
            edge_index_unique[:,1:2].double(), 
            edge_attr_unique[:,2:4].double()
            ), dim=1)
    edge_endpoints = torch.cat((sources, dests), dim=0) # (2E, 3/4)
    
    # get unique pins
    pin_info, pin_inverse_index = torch.unique(edge_endpoints, return_inverse=True, dim=0) # (E_u, 3), (2E)
    pin_edge_index = pin_inverse_index.view(2, E//2)
    pin_map = pin_info[:, 0].type(cond.edge_index.dtype)
    pin_offsets = pin_info[:, 1:3].type(cond.edge_attr.dtype)
    return pin_map, pin_offsets, pin_edge_index
    