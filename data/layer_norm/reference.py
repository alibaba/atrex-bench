import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return x.clone()
        return F.layer_norm(x.float(), (x.shape[-1],), weight=weight.float(), bias=bias.float()).to(x.dtype)
