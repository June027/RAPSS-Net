import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Batch-wise supervised contrastive loss for classification features."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features is None or labels is None:
            device = features.device if isinstance(features, torch.Tensor) else None
            return torch.zeros((), device=device)

        if features.dim() > 2:
            features = features.reshape(features.size(0), -1)
        if features.dim() != 2 or features.size(0) < 2:
            return features.new_zeros(1, requires_grad=True).squeeze()

        labels = labels.view(-1)
        if labels.numel() != features.size(0):
            raise ValueError("features and labels batch size mismatch.")

        features = F.normalize(features.float(), dim=1, p=2)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        batch_size = features.size(0)
        eye_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        positive_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)) & (~eye_mask)
        if not positive_mask.any():
            return features.new_zeros(1, requires_grad=True).squeeze()

        negative_inf = torch.tensor(-1.0e9, device=features.device, dtype=logits.dtype)
        masked_logits = torch.where(eye_mask, negative_inf, logits)
        log_prob = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)

        positive_counts = positive_mask.sum(dim=1)
        valid_anchors = positive_counts > 0
        mean_log_prob_pos = (
            (log_prob * positive_mask.float()).sum(dim=1)
            / positive_counts.clamp_min(1).float()
        )
        return -mean_log_prob_pos[valid_anchors].mean()
