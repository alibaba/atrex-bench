import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, hidden_size: int, eps: float=1e-06) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        norm_sq = (x_float * x_float).sum(dim=-1, keepdim=True)
        return (x_float * torch.rsqrt(norm_sq + self.eps)).to(x.dtype)
