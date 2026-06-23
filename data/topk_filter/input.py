import torch


def _make_inputs(seq_len: int, hidden_size: int) -> dict[str, torch.Tensor]:
    logits = torch.randn(seq_len, hidden_size, dtype=torch.float32, device='cuda')
    return {'logits': logits}
