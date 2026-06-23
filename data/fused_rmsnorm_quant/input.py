import torch


def _make_inputs(token_count: int, hidden_size: int) -> dict[str, torch.Tensor]:
    x = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device='cuda')
    residual = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device='cuda')
    return {'x': x, 'residual': residual, 'weight': weight}
