import torch

def _make_inputs(token_count: int, num_experts: int) -> dict[str, torch.Tensor]:
    gating_output = torch.randn(token_count, num_experts, dtype=torch.float32, device='cuda')
    return {'gating_output': gating_output}
