import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    """Minimal PyTorch reference Mamba layer for inference on Windows.

    The parameter layout matches mamba-ssm 1.x for checkpoints used by this
    repository. It intentionally avoids CUDA extension dependencies.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        bias=False,
        conv_bias=True,
        **_,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (
            math.ceil(self.d_model / 16)
            if dt_rank == "auto"
            else int(dt_rank)
        )

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, hidden_states, inference_params=None):
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Mamba expects (batch, seqlen, dim), got {tuple(hidden_states.shape)}"
            )
        batch, seqlen, _ = hidden_states.shape
        input_dtype = hidden_states.dtype

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2).contiguous()
        x = self.conv1d(x)[..., :seqlen]
        x = F.silu(x)

        x_flat = x.transpose(1, 2).reshape(batch * seqlen, self.d_inner)
        x_db = self.x_proj(x_flat)
        dt, b_param, c_param = torch.split(
            x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = F.linear(dt, self.dt_proj.weight)
        dt = dt.reshape(batch, seqlen, self.d_inner).transpose(1, 2).contiguous()
        b_param = b_param.reshape(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        c_param = c_param.reshape(batch, seqlen, self.d_state).transpose(1, 2).contiguous()

        y = self._selective_scan_ref(
            x,
            dt,
            b_param,
            c_param,
            z.transpose(1, 2).contiguous(),
            input_dtype,
        )
        y = y.transpose(1, 2).contiguous()
        return self.out_proj(y)

    def _selective_scan_ref(self, u, delta, b_param, c_param, z, output_dtype):
        u_float = u.float()
        delta = delta.float()
        delta = F.softplus(delta + self.dt_proj.bias.float().view(1, -1, 1))
        a = -torch.exp(self.A_log.float())
        d_skip = self.D.float().view(1, -1)

        batch, dim, seqlen = u_float.shape
        state = torch.zeros(
            batch,
            dim,
            self.d_state,
            dtype=torch.float32,
            device=u.device,
        )
        outputs = []
        for t in range(seqlen):
            delta_t = delta[:, :, t]
            u_t = u_float[:, :, t]
            b_t = b_param[:, :, t].float()
            c_t = c_param[:, :, t].float()
            delta_a = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
            delta_b_u = (
                delta_t.unsqueeze(-1)
                * b_t.unsqueeze(1)
                * u_t.unsqueeze(-1)
            )
            state = delta_a * state + delta_b_u
            y_t = torch.einsum("bdn,bn->bd", state, c_t)
            y_t = y_t + u_t * d_skip
            y_t = y_t * F.silu(z[:, :, t].float())
            outputs.append(y_t)
        return torch.stack(outputs, dim=2).to(output_dtype)
