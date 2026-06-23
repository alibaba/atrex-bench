import torch

def _make_inputs(num_tokens: int, head_num: int, kv_head_num: int, size_per_head: int=128, dtype: str='fp16') -> dict[str, torch.Tensor]:
    dt = torch.float16 if dtype == 'fp16' else torch.bfloat16
    hidden_size = head_num * size_per_head + 2 * kv_head_num * size_per_head
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dt, device='cuda')
    q_weight = torch.randn(size_per_head, dtype=dt, device='cuda')
    k_weight = torch.randn(size_per_head, dtype=dt, device='cuda')
    return {'hidden_states': hidden_states, 'q_weight': q_weight, 'k_weight': k_weight}
