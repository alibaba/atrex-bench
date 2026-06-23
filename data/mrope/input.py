import torch

def _make_inputs(token_count: int, num_query_heads: int, num_kv_heads: int, head_size: int, rotary_dim: int) -> dict[str, torch.Tensor]:
    q = torch.randn(token_count, num_query_heads * head_size, dtype=torch.bfloat16, device='cuda') * 0.02
    k = torch.randn(token_count, num_kv_heads * head_size, dtype=torch.bfloat16, device='cuda') * 0.02
    half = rotary_dim // 2
    angles = torch.randn(3, token_count, half, dtype=torch.float32, device='cuda') * 0.1
    return {'q': q, 'k': k, 'cos': torch.cos(angles), 'sin': torch.sin(angles)}
