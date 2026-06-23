import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, top_k: int=50) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        top_k = min(self.top_k, logits.shape[-1])
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < threshold, float('-inf'))
