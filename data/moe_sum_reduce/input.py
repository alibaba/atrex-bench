import torch

def _make_inputs(token_count: int, top_k: int, hidden_size: int) -> dict[str, torch.Tensor]:
    expert_outputs = torch.randn(token_count, top_k, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02
    return {'expert_outputs': expert_outputs}
