import torch

def _make_inputs(rows: int, hidden_size: int) -> dict[str, torch.Tensor]:
    x = torch.randn(rows, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02 + 1.0
    z = torch.randn_like(x)
    return {'x': x, 'weight': weight, 'z': z}
