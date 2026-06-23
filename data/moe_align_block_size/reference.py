import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, block_size: int, num_experts: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.num_experts = num_experts

    def forward(self, topk_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        flat_ids = topk_ids.flatten()
        total_tokens = flat_ids.numel()
        sorted_chunks = []
        expert_blocks = []
        for expert in range(self.num_experts):
            token_ids = torch.nonzero(flat_ids == expert, as_tuple=False).flatten().to(torch.int32)
            if token_ids.numel() == 0:
                continue
            pad = -token_ids.numel() % self.block_size
            if pad:
                padding = torch.full((pad,), total_tokens, dtype=torch.int32, device=topk_ids.device)
                token_ids = torch.cat([token_ids, padding])
            sorted_chunks.append(token_ids)
            expert_blocks.extend([expert] * (token_ids.numel() // self.block_size))
        if sorted_chunks:
            sorted_token_ids = torch.cat(sorted_chunks)
        else:
            sorted_token_ids = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
        expert_ids = torch.tensor(expert_blocks, dtype=torch.int32, device=topk_ids.device)
        num_tokens_post_pad = torch.tensor([sorted_token_ids.numel()], dtype=torch.int32, device=topk_ids.device)
        return {'sorted_token_ids': sorted_token_ids, 'expert_ids': expert_ids, 'num_tokens_post_pad': num_tokens_post_pad}
