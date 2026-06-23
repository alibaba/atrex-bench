import torch

def _make_inputs(token_count: int, hidden_size: int, out_features: int) -> dict[str, torch.Tensor]:
    hidden_states = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02
    weight = torch.randn(out_features, hidden_size, dtype=torch.bfloat16, device='cuda') * hidden_size ** (-0.5)
    bias = torch.randn(out_features, dtype=torch.bfloat16, device='cuda') * 0.02
    post_mul_mat = torch.randn(token_count, out_features, dtype=torch.bfloat16, device='cuda') * 0.02
    return {'hidden_states': hidden_states, 'weight': weight, 'bias': bias, 'post_mul_mat': post_mul_mat}
