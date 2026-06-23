import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, num_experts: int, intermediate_size: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        intermediate_size = self.intermediate_size
        output = torch.zeros(token_count, hidden_size, device=hidden_states.device, dtype=torch.float32)
        hidden_fp32 = hidden_states.float()
        for expert_index in range(self.num_experts):
            mask = topk_ids == expert_index
            if not mask.any():
                continue
            weight_for_expert = (topk_weights * mask.to(topk_weights.dtype)).sum(dim=1)
            token_idx = weight_for_expert.nonzero(as_tuple=True)[0]
            if token_idx.numel() == 0:
                continue
            x = hidden_fp32.index_select(0, token_idx)
            w1_e = w1[expert_index].float()
            w2_e = w2[expert_index].float()
            intermediate = F.linear(x, w1_e)
            gate = intermediate[:, :intermediate_size]
            up = intermediate[:, intermediate_size:]
            activated = F.silu(gate) * up
            expert_output = F.linear(activated, w2_e)
            output.index_add_(0, token_idx, expert_output * weight_for_expert.index_select(0, token_idx).unsqueeze(1))
        return output
