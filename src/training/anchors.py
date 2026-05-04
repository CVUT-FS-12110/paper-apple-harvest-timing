import torch
import torch.nn.functional as F


def make_regular_anchor_centers_days(
    day_min: int = -30,
    day_max: int = 100,
    step: int = 10,
    scale: float = 100.0,
) -> torch.Tensor:
    """
    Returns anchor centers in the SAME scaled units as dataset targets.
    Example: day=-30 becomes -0.30 when scale=100.
    """
    days = list(range(day_min, day_max + 1, step))
    centers = torch.tensor(days, dtype=torch.float32) / scale
    return centers


def make_observed_anchor_centers_days(
    observed_days: list[int],
    scale: float = 100.0,
) -> torch.Tensor:
    """
    Alternative: use unique observed target days as anchors.
    """
    unique_days = sorted(set(observed_days))
    centers = torch.tensor(unique_days, dtype=torch.float32) / scale
    return centers


def gaussian_anchor_targets(
    targets: torch.Tensor,
    anchor_centers: torch.Tensor,
    sigma_scaled: float,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    targets: [B] scaled day targets
    anchor_centers: [K] scaled anchor centers
    returns:
        if normalize=True: [B, K] soft target distributions summing to 1
        else: [B, K] unnormalized Gaussian memberships
    """
    if targets.ndim != 1:
        targets = targets.view(-1)

    diff = targets[:, None] - anchor_centers[None, :]
    soft = torch.exp(-(diff ** 2) / (2.0 * sigma_scaled ** 2))

    if normalize:
        soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(eps)

    return soft


def decode_anchor_logits_to_scalar(
    logits: torch.Tensor,
    anchor_centers: torch.Tensor,
) -> torch.Tensor:
    """
    logits: [B, K]
    anchor_centers: [K]
    returns: [B] decoded scalar prediction in scaled units

    Softmax distribution decoding:
        pred = sum_k p_k * c_k
    """
    probs = F.softmax(logits, dim=1)
    pred = (probs * anchor_centers[None, :]).sum(dim=1)
    return pred


def decode_anchor_probs_to_scalar(
    probs: torch.Tensor,
    anchor_centers: torch.Tensor,
) -> torch.Tensor:
    """
    Same as above, but if you already have normalized probabilities.
    """
    pred = (probs * anchor_centers[None, :]).sum(dim=1)
    return pred