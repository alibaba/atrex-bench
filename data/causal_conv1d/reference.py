import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, activation: str | None='silu') -> None:
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        seq = x.unsqueeze(0).to(weight.dtype)
        state = initial_state.to(weight.dtype)
        width = weight.shape[1]
        seq_with_state = torch.cat([state[:, :, -(width - 1):], seq], dim=-1)
        out = F.conv1d(seq_with_state, weight.unsqueeze(1), bias, groups=weight.shape[0])
        out = out[:, :, -x.shape[-1]:]
        if self.activation in ('silu', 'swish'):
            out = F.silu(out)
        elif self.activation is not None:
            raise ValueError(f'unsupported activation: {self.activation}')
        return out.squeeze(0).to(dtype)
