import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, eps: float=1e-05) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_out = x + residual
        normed = F.rms_norm(residual_out, (residual_out.shape[-1],), weight, self.eps)
        normed_f = normed.to(torch.float32)
        finfo = torch.finfo(torch.float8_e4m3fnuz)
        amax = normed_f.abs().amax(dim=-1, keepdim=True)
        scale = amax / finfo.max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = (normed_f / scale).to(torch.float8_e4m3fnuz)
        return (quantized, scale.to(torch.float32), residual_out)
