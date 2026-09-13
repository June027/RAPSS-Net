import torch
import torch.nn as nn

from .contrastive_loss import SupervisedContrastiveLoss
from .dice_loss import DiceCELoss
from .domain_debiasing_loss import DomainDebiasingLoss
from .focal_loss import FocalLoss
from .habitat_alignment_loss import HabitatAlignmentLoss


def _build_classification_loss(config, class_counts, device):
    loss_name = str(config.get("train", {}).get("classification_loss", "ce")).lower()
    label_smooth = float(config.get("train", {}).get("label_smoothing", 0.0))
    use_class_weights = bool(config.get("train", {}).get("use_class_weights", True))
    weight_tensor = None
    if use_class_weights and class_counts is not None and len(class_counts) == 2:
        neg_count, pos_count = [int(x) for x in class_counts]
        if neg_count > 0 and pos_count > 0:
            total = float(neg_count + pos_count)
            weights = [total / (2.0 * neg_count), total / (2.0 * pos_count)]
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smooth, weight=weight_tensor)
    if loss_name == "focal":
        focal_gamma = float(config.get("train", {}).get("focal_gamma", 2.0))
        return FocalLoss(gamma=focal_gamma, weight=weight_tensor)
    raise ValueError(f"Unsupported train.classification_loss: {loss_name}")


def _build_seg_pos_weight(config, device):
    seg_pos_weight = config.get("train", {}).get("seg_pos_weight", None)
    if seg_pos_weight is None:
        return None
    seg_pos_weight = float(seg_pos_weight)
    if seg_pos_weight <= 0:
        return None
    return torch.tensor([seg_pos_weight], dtype=torch.float32, device=device)


def build_losses(config, class_counts, device):
    seg_pos_weight = _build_seg_pos_weight(config, device)
    criterions = {
        "pcr": _build_classification_loss(config, class_counts, device),
        "seg": DiceCELoss(pos_weight=seg_pos_weight),
    }

    use_domain_debias = bool(config.get("train", {}).get("use_domain_debias", False))
    if use_domain_debias:
        criterions["domain"] = DomainDebiasingLoss()

    contrastive_weight = float(config.get("loss_weights", {}).get("contrastive", 0.0))
    if contrastive_weight > 0:
        temperature = float(config.get("train", {}).get("contrastive_temperature", 0.2))
        criterions["contrastive"] = SupervisedContrastiveLoss(
            temperature=temperature
        )

    habitat_weight = float(config.get("loss_weights", {}).get("habitat", 0.0))
    if habitat_weight > 0:
        habitat_cfg = config.get("train", {})
        criterions["habitat"] = HabitatAlignmentLoss(
            erosion_kernel_size=int(habitat_cfg.get("habitat_erosion_kernel", 3)),
            dilation_kernel_size=int(habitat_cfg.get("habitat_dilation_kernel", 5)),
        )

    return criterions
