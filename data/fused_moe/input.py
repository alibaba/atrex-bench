import torch

def _make_inputs(token_count: int, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int) -> dict[str, torch.Tensor]:
    hidden_states = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.1
    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device='cuda') * hidden_size ** (-0.5)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device='cuda') * intermediate_size ** (-0.5)
    topk_weights = torch.softmax(torch.randn(token_count, top_k, dtype=torch.float32, device='cuda'), dim=-1)
    topk_ids = torch.randint(0, num_experts, (token_count, top_k), dtype=torch.int32, device='cuda')
    return {'hidden_states': hidden_states, 'w1': w1, 'w2': w2, 'topk_weights': topk_weights, 'topk_ids': topk_ids}
