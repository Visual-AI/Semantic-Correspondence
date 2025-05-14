import torch


def expand_size(coord, size) -> torch.Tensor:
    """
    Expand 'size' to match coord's dimensions for per-batch operations.
    
    Args:
        coord (torch.Tensor): Coordinates of shape (B, ..., 2) where last dimension is (x, y).
        size (Union[Tuple[int, int], torch.Tensor]): Image size as (H, W) or tensor of shape (B, 2).
    
    Returns:
        torch.Tensor: Expanded size of shape (B, ..., 2) matching coord's dimensions.
    """
    B, ndim = coord.shape[0], coord.ndim
    device = coord.device

    if isinstance(size, tuple):
        size = torch.tensor(size, device=device).expand(B, -1)
    
    return size.view(B, *((1,) * (ndim - 2)), 2)


def normalize_coordinates(coord, size) -> torch.Tensor:
    """
    Normalize coordinates from size scale to the range (-1, 1).

    Args:
        coord (torch.Tensor): Coordinates of shape (B, ..., 2) where last dimension is (x, y).
        size (Union[Tuple[int, int], torch.Tensor]): Image size as (H, W) or tensor of shape (B, 2).

    Returns:
        torch.Tensor: Normalized coordinates in range (-1, 1) with same shape as input coord.
    """
    size = expand_size(coord, size).flip(-1)  # flip (H, W) to (W, H)
    return 2 * coord / (size - 1) - 1


def unnormalize_coordinates(coord, size) -> torch.Tensor:
    """
    Unnormalize coordinates from range (-1, 1) to size scale.

    Args:
        coord (torch.Tensor): Normalized coordinates of shape (B, ..., 2) where last dimension is (x, y).
        size (Union[Tuple[int, int], torch.Tensor]): Image size as (H, W) or tensor of shape (B, 2).

    Returns:
        torch.Tensor: Unnormalized coordinates in size scale with same shape as input coord.
    """
    size = expand_size(coord, size).flip(-1)  # flip (H, W) to (W, H)
    return ((coord + 1) / 2) * (size - 1)


def scaling_coordinates(coord, src_scale, trg_scale, mode='align_corner') -> torch.Tensor:
    """
    Scale coordinates from src_scale to trg_scale.

    Args:
        coord (torch.Tensor): (B, ..., 2) coordinates (x, y) in src_scale.
        src_scale (Union[Tuple[int, int], torch.Tensor]): Source scale (H1, W1) or (B, 2).
        trg_scale (Union[Tuple[int, int], torch.Tensor]): Target scale (H2, W2) or (B, 2).
        mode (str): 'align_corner' (default) or 'simple' or 'center'.
            - align_corner: four corners always mapped as corner.
            - simple: directly scale coordinates by the ratio of target to source scale.
            - center: coord at smaller scale treated as center of squared patch of larger scale.
            
    Returns:
        torch.Tensor: (B, ..., 2) coordinates (x, y) in trg_scale.
    """
    if mode not in ['simple', 'align_corner', 'center']:
        raise ValueError(f"Invalid mode: {mode}. Use 'align_corner', 'simple', or 'center'.")

    src_scale = expand_size(coord, src_scale).flip(-1)  # (B, ..., 2) as (W1, H1)
    trg_scale = expand_size(coord, trg_scale).flip(-1)  # (B, ..., 2) as (W2, H2)

    if mode == 'align_corner':
        return coord / (src_scale - 1) * (trg_scale - 1)
    elif mode == 'simple':
        return coord / src_scale * trg_scale
    else:  # center mode
        return (coord + 0.5) / src_scale * trg_scale - 0.5
    

def regularize_coordinates(coord, size, eps = 0.) -> torch.Tensor:
    """
    Clamp coordinates to image bounds.

    Args:
        coord (torch.Tensor): (B, ..., 2) coordinates (x, y).
        size (Union[Tuple[int, int], torch.Tensor]): Image size as (H, W) or tensor of shape (B, 2).
        eps (float): Small offset for clamping bounds. Default: 0.

    Returns:
        torch.Tensor: (B, ..., 2) Clamped coordinates.
    """
    size = expand_size(coord, size).flip(-1) - 1  # Convert (H, W) to (W-1, H-1)
    min_bound = torch.full_like(coord, eps)
    max_bound = size - eps
    return torch.clamp(coord, min=min_bound, max=max_bound).to(coord)


def create_grid(H, W, step=1, device='cpu') -> torch.Tensor:
    """
    Create a 2D grid of (x, y) coordinates.

    Args:
        H (int): Height of the grid.
        W (int): Width of the grid.
        step (int): Spacing between grid points (default: 1).
        device (str): Torch device for the grid (default: 'cpu').

    Returns:
        torch.Tensor: Shape (H, W, 2) containing (x, y) coordinates.
    """
    x = torch.arange(0, W, step, device=device)
    y = torch.arange(0, H, step, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    return torch.stack((x_grid, y_grid), dim=2)