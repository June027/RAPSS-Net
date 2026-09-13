import torch
import torch.nn as nn


class IBN3D(nn.Module):
    """
    Instance-Batch Normalization (IBN) for 3D feature maps.

    Splits channels into two halves:
      - First half uses InstanceNorm3d to suppress MRI scanner domain shift
        (style normalization across acquisition protocols).
      - Second half uses BatchNorm3d to preserve tumor pathological texture
        and semantic content features.
    This design enables robust cross-center generalization without discarding
    diagnostically relevant local statistics.
    """

    def __init__(self, out_dim):
        super(IBN3D, self).__init__()
        half1 = out_dim // 2
        half2 = out_dim - half1
        self.IN = nn.InstanceNorm3d(half1, affine=True)
        self.BN = nn.BatchNorm3d(half2)

    def forward(self, x):
        split = torch.split(x, [self.IN.num_features, self.BN.num_features], dim=1)
        out1 = self.IN(split[0])
        out2 = self.BN(split[1])
        return torch.cat((out1, out2), dim=1)


class ConvBlock3D(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        # IBN replaces standard normalization to jointly mitigate domain shift and preserve content
        self.norm1 = IBN3D(out_dim)

        self.conv2 = nn.Conv3d(
            out_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # Second IBN layer further consolidates domain-invariant representations
        self.norm2 = IBN3D(out_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.downsample = None
        if stride != 1 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_dim),  # Linear projection: standard BN is sufficient
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return self.act(out)


class UpConvBlock3D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UpConvBlock3D, self).__init__()
        self.upconv = nn.ConvTranspose3d(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        # Decoder targets spatial reconstruction; InstanceNorm preserves fine-grained texture
        self.norm = nn.InstanceNorm3d(out_dim, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.upconv(x)))


class GlobalFeatureRecalibration(nn.Module):
    """
    Lightweight squeeze-excitation style recalibration for pooled feature vectors.

    This block is intentionally small so it can be attached to the paper-facing
    TriDual3D classifier head without perturbing the rest of the training stack.
    """

    def __init__(self, dim: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive for GlobalFeatureRecalibration.")
        if reduction <= 0:
            raise ValueError("reduction must be positive for GlobalFeatureRecalibration.")
        hidden_dim = max(dim // reduction, 16)
        self.pre_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(self.pre_norm(x))
        return self.out_norm(x * (1.0 + gate))
