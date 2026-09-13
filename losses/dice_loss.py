import torch
import torch.nn as nn


class DiceCELoss(nn.Module):
    """
    Dice Loss + Binary Cross Entropy Loss
    用于辅助分割分支，强迫网络精准识别肿瘤的三维解剖结构。

    参数:
        dice_weight: Dice Loss 权重
        ce_weight: BCE Loss 权重
        smooth: Dice 数值稳定项
        pos_weight: BCE 正样本权重张量 (torch.Tensor 或 None)。
                    数据不平衡时传入 neg_voxels / pos_voxels 比值，
                    放大正（肿瘤）体素的梯度贡献，与分类头 class_weights 策略一致。
    """

    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-5, pos_weight=None):
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        # pos_weight 在 BCEWithLogitsLoss 内部直接广播到 (1,) 张量
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        # logits: [B, 1, D, H, W]
        # targets: [B, 1, D, H, W]

        # 1. BCE Loss（含 pos_weight 补偿不平衡）
        ce_loss = self.bce(logits, targets)

        # 2. Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        # 综合损失
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss
