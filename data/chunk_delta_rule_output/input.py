import math
import torch

def _make_inputs(batch_size: int, token_count: int, num_q_heads: int, num_v_heads: int, key_dim: int, value_dim: int, chunk_size: int=64) -> dict[str, torch.Tensor]:
    q = torch.randn(batch_size, token_count, num_q_heads, key_dim, dtype=torch.bfloat16, device='cuda') * 0.02
    k = torch.randn_like(q)
    v = torch.randn(batch_size, token_count, num_v_heads, value_dim, dtype=torch.bfloat16, device='cuda') * 0.02
    num_chunks = math.ceil(token_count / chunk_size)
    h = torch.randn(batch_size, num_chunks, num_v_heads, key_dim, value_dim, dtype=torch.bfloat16, device='cuda') * 0.02
    g = -torch.rand(batch_size, token_count, num_v_heads, dtype=torch.float32, device='cuda')
    return {'q': q, 'k': k, 'v': v, 'h': h, 'g': g}
