import torch

def _make_inputs(token_count: int, dim: int, width: int=4) -> dict[str, torch.Tensor]:
    x = torch.randn(dim, token_count, dtype=torch.bfloat16, device='cuda') * 0.02
    weight = torch.randn(dim, width, dtype=torch.bfloat16, device='cuda') * 0.02
    bias = torch.randn(dim, dtype=torch.bfloat16, device='cuda') * 0.02
    initial_state = torch.randn(1, dim, width - 1, dtype=torch.bfloat16, device='cuda') * 0.02
    return {'x': x, 'weight': weight, 'bias': bias, 'initial_state': initial_state}
