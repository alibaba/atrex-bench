import torch
import torch.nn as nn

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class Model(nn.Module):

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, rope_dim: int, rope_base: float=10000.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.rope_base = rope_base

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = self.rope_dim
        x_rope = x[..., :d]
        x_pass = x[..., d:]
        rotated = x_rope.float() * cos + _rotate_half(x_rope.float()) * sin
        rotated = rotated.to(x.dtype)
        if d < self.head_dim:
            return torch.cat([rotated, x_pass], dim=-1)
        return rotated

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor, qkv_bias: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.num_heads * self.head_dim
        kv_n = self.num_kv_heads * self.head_dim
        if qkv_bias is not None:
            qkv = qkv + qkv_bias
        q = qkv[..., :n].reshape(-1, self.num_heads, self.head_dim)
        k = qkv[..., n:n + kv_n].reshape(-1, self.num_kv_heads, self.head_dim)
        v = qkv[..., n + kv_n:n + 2 * kv_n].reshape(-1, self.num_kv_heads, self.head_dim)
        inv_freq = 1.0 / self.rope_base ** (torch.arange(0, self.rope_dim, 2, device=qkv.device, dtype=torch.float32) / self.rope_dim)
        freqs = positions.unsqueeze(-1).float() * inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        return (q, k, v)
