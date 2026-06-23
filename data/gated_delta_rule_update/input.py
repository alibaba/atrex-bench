import torch

def _make_inputs(batch_size: int, token_count: int, num_q_heads: int, num_v_heads: int, key_dim: int, value_dim: int, state_count: int) -> dict[str, torch.Tensor]:
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device='cuda') * 0.02
    a = torch.randn(batch_size, token_count, num_v_heads, dtype=torch.bfloat16, device='cuda') * 0.02
    dt_bias = torch.randn(num_v_heads, dtype=torch.bfloat16, device='cuda') * 0.02
    q = torch.randn(batch_size, token_count, num_q_heads, key_dim, dtype=torch.bfloat16, device='cuda') * 0.02
    k = torch.randn_like(q)
    v = torch.randn(batch_size, token_count, num_v_heads, value_dim, dtype=torch.bfloat16, device='cuda') * 0.02
    b = torch.randn_like(a)
    initial_state = torch.randn(state_count, num_v_heads, key_dim, value_dim, dtype=torch.float32, device='cuda') * 0.02
    initial_state_indices = torch.arange(batch_size, dtype=torch.int32, device='cuda') % state_count
    return {'A_log': A_log, 'a': a, 'dt_bias': dt_bias, 'q': q, 'k': k, 'v': v, 'b': b, 'initial_state': initial_state, 'initial_state_indices': initial_state_indices}
