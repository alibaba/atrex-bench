import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] >= 2:
            split = x.shape[-1] // 2
            gate = x[..., :split]
            up = x[..., split:split + split]
            activated = F.silu(gate.float()).to(x.dtype)
            return (activated * up).to(x.dtype)
        return F.silu(x.float()).to(x.dtype)
