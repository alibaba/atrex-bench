import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, eps: float=1e-06, norm_before_gate: bool=True, activation: str='silu') -> None:
        super().__init__()
        self.eps = float(eps)
        self.norm_before_gate = bool(norm_before_gate)
        self.activation = activation

    def forward(self, x: torch.Tensor, weight: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        z_f = z.float()
        if not self.norm_before_gate:
            x_f = x_f * self._activate(z_f)
        rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + self.eps)
        out = x_f * rstd * weight.float()
        if self.norm_before_gate:
            out = out * self._activate(z_f)
        return out.to(dtype)

    def _activate(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation in ('silu', 'swish'):
            return F.silu(z)
        if self.activation == 'sigmoid':
            return torch.sigmoid(z)
        raise ValueError(f'unsupported activation: {self.activation}')
