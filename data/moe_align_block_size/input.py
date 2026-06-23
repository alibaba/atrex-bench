import torch

def _make_inputs(token_count: int, top_k: int, num_experts: int) -> dict[str, torch.Tensor]:
    topk_ids = torch.randint(0, num_experts, (token_count, top_k), dtype=torch.int32, device='cuda')
    return {'topk_ids': topk_ids}
