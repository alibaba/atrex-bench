import torch

def _make_inputs(batch_size: int, token_count: int, num_k_heads: int, num_v_heads: int, key_dim: int, value_dim: int, state_count: int) -> dict[str, torch.Tensor]:
    k = torch.randn(batch_size, token_count, num_k_heads, key_dim, dtype=torch.bfloat16, device='cuda') * 0.02
    w = torch.randn(batch_size, token_count, num_v_heads, key_dim, dtype=torch.bfloat16, device='cuda') * 0.02
    u = torch.randn(batch_size, token_count, num_v_heads, value_dim, dtype=torch.bfloat16, device='cuda') * 0.02
    g = -torch.rand(batch_size, token_count, num_v_heads, dtype=torch.float32, device='cuda')
    initial_state = torch.randn(state_count, num_v_heads, key_dim, value_dim, dtype=torch.float32, device='cuda') * 0.02
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device='cuda') % state_count
    return {'k': k, 'w': w, 'u': u, 'g': g, 'initial_state': initial_state, 'initial_state_indices': initial_state_indices}
