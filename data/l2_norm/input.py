import torch

def _make_inputs(num_tokens: int, hidden_size: int, dtype: str='bfloat16') -> dict[str, torch.Tensor]:
    dt = getattr(torch, dtype)
    return {'x': torch.randn(num_tokens, hidden_size, dtype=dt, device='cuda')}

def get_inputs() -> list[torch.Tensor]:
    return list(_make_inputs(num_tokens=1024, hidden_size=128).values())

def get_init_inputs() -> dict[str, object]:
    return {'hidden_size': 128}
