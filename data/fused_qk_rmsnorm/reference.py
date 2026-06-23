import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, head_num: int, kv_head_num: int, size_per_head: int=128, eps: float=1e-06) -> None:
        super().__init__()
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.eps = eps
        self.q_size = head_num * size_per_head
        self.kv_size = kv_head_num * size_per_head

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return weight * x.to(input_dtype)

    def forward(self, hidden_states: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor) -> torch.Tensor:
        (q, k, v) = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self._rmsnorm(q.reshape(-1, self.size_per_head), q_weight).view(q.shape)
        k = self._rmsnorm(k.reshape(-1, self.size_per_head), k_weight).view(k.shape)
        return torch.cat([q, k, v], dim=-1)
