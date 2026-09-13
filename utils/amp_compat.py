import torch


def build_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler("cuda", enabled=enabled)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    raise AttributeError("No compatible GradScaler implementation found in this PyTorch build.")


def autocast_context(enabled, dtype):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled, dtype=dtype)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    raise AttributeError("No compatible autocast implementation found in this PyTorch build.")
