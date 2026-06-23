import torch

def _make_inputs(token_count: int, hidden_size: int) -> dict[str, torch.Tensor]:
    x = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02 + 1.0
    return {'x': x, 'weight': weight}
