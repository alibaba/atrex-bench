import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, routed_scaling_factor: float=1.0) -> None:
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        out = expert_outputs.float().sum(dim=1) * self.routed_scaling_factor
        return out.to(expert_outputs.dtype)
