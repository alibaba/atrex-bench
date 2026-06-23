import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, eps: float=1e-06) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        residual_out = (x + residual).to(residual.dtype)
        residual_float = residual_out.float()
        variance = residual_float.square().mean(dim=-1, keepdim=True)
        normed = residual_float * torch.rsqrt(variance + self.eps) * weight.float()
        return {'out': normed.to(x.dtype), 'residual': residual_out}
