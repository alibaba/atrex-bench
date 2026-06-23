import torch

def _make_inputs(total_tokens: int, num_heads: int, num_kv_heads: int, head_dim: int, with_bias: bool=False, dtype: str='bf16') -> dict[str, torch.Tensor]:
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    n = num_heads * head_dim
    kv_n = num_kv_heads * head_dim
    qkv_dim = n + 2 * kv_n
    qkv = torch.randn(total_tokens, qkv_dim, dtype=dt, device='cuda')
    positions = torch.arange(total_tokens, dtype=torch.int32, device='cuda')
    result: dict[str, torch.Tensor] = {'qkv': qkv, 'positions': positions}
    if with_bias:
        result['qkv_bias'] = torch.randn(qkv_dim, dtype=dt, device='cuda') * 0.01
    return result
