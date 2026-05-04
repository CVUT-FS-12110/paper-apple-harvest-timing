import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy with soft targets.

    logits: [B, K]
    soft_targets: [B, K], each row sums to 1
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss