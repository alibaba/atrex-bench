import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, post_mul_mat: torch.Tensor) -> torch.Tensor:
        gate = F.linear(hidden_states, weight, bias)
        out = torch.sigmoid(gate.float()) * post_mul_mat.float()
        return out.to(post_mul_mat.dtype)
