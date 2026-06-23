import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, eps: float=1e-06) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        variance = x_f.square().mean(dim=-1, keepdim=True)
        out = x_f * torch.rsqrt(variance + self.eps) * weight.float()
        return out.to(x.dtype)
