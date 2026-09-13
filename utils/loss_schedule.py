from __future__ import annotations


def compute_aux_loss_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(max(float(epoch), 0.0) / float(warmup_epochs), 1.0)
