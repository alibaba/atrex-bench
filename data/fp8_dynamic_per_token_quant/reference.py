import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.finfo = torch.finfo(torch.float8_e4m3fnuz)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_float = x.to(torch.float32)
        row_absmax = x_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = row_absmax / self.finfo.max
        inv_scale = scale.reciprocal()
        q = (x_float * inv_scale).clamp(min=self.finfo.min, max=self.finfo.max)
        return (q.to(torch.float8_e4m3fnuz), scale.squeeze(-1))
