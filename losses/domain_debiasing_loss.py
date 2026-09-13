import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDebiasingLoss(nn.Module):
    """
    Cross-Domain Supervised Contrastive Loss (Cross-Domain SupCon).
    It aligns features of the same class across different centers (pull together),
    while separating features of different classes (push apart).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        center_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates cross-domain contrastive loss.
        features: Tensor of shape (B, D) or (B, C, H, W, D).
        center_ids: Tensor of shape (B,) containing center indices.
        labels: Tensor of shape (B,) containing class labels.
        """
        if features.dim() > 2:
            features = F.adaptive_avg_pool3d(features, 1).view(features.size(0), -1)

        if features.dim() != 2 or features.size(0) < 2:
            # AMP-safe 梯度钉子: features.sum()*0 在 fp16/inf 下为 NaN，改用 new_zeros 确保安全
            return features.new_zeros(1, requires_grad=True).squeeze()

        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature

        batch_size = features.size(0)
        center_ids = center_ids.view(-1, 1)
        labels = labels.view(-1, 1)
        if center_ids.size(0) != batch_size or labels.size(0) != batch_size:
            raise ValueError("features, center_ids, labels batch size mismatch.")

        eye_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        same_center = torch.eq(center_ids, center_ids.T)
        same_label = torch.eq(labels, labels.T)

        pos_mask = same_label & (~same_center) & (~eye_mask)
        neg_mask = (~same_label) & (~eye_mask)
        valid_mask = pos_mask | neg_mask

        has_positive = pos_mask.any(dim=1)
        if not has_positive.any():
            return features.new_zeros((), requires_grad=True)
            # AMP-safe 梯度钉子，确保 DDP 计算图连通
            return features.new_zeros(1, requires_grad=True).squeeze()

        # To resolve logsumexp numerical explosions, we mask invalid pairs with a safe minimum (-1e9).
        # We explicitly mask the diagonal (self-similarity) and invalid contrastive pairs.
        masked_logits = logits.clone()
        masked_logits[eye_mask] = -1e9  # Exclude self-similarity

        # Apply valid mask for remaining cross-domain logic
        masked_logits = torch.where(
            valid_mask,
            masked_logits,
            torch.tensor(-1e9, device=features.device, dtype=logits.dtype)
        )
        log_prob = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
        mean_log_prob_pos = (
            (log_prob * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp_min(1.0)
        )
        return -mean_log_prob_pos[has_positive].mean()
