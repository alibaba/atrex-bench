import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_experts: int=128, block_size: int=128) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.block_size = int(block_size)

    def forward(self, topk_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        flat_ids = topk_ids.flatten()
        counts = torch.bincount(flat_ids.to(torch.long), minlength=self.num_experts)[:self.num_experts].to(torch.int32)
        cumsum = torch.empty(self.num_experts + 1, dtype=torch.int32, device=topk_ids.device)
        cumsum[0] = 0
        cumsum[1:] = torch.cumsum(counts, dim=0)
        sorted_chunks = []
        for expert in range(self.num_experts):
            token_ids = torch.nonzero(flat_ids == expert, as_tuple=False).flatten().to(torch.int32)
            if token_ids.numel():
                sorted_chunks.append(token_ids)
        if sorted_chunks:
            sorted_token_ids = torch.cat(sorted_chunks)
        else:
            sorted_token_ids = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
        return {'sorted_token_ids': sorted_token_ids, 'expert_token_counts': counts, 'cumsum': cumsum}
