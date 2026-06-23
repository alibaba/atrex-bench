import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, num_experts: int, intermediate_size: int, top_k: int, scale_block_n: int=128, scale_block_k: int=128) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.scale_block_n = scale_block_n
        self.scale_block_k = scale_block_k

    def forward(self, hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, a_scale: torch.Tensor, fc1_scale: torch.Tensor, fc2_scale: torch.Tensor) -> torch.Tensor:
        (B, D) = hidden_states.shape
        top_k = topk_ids.shape[1]
        (expert, model_dim, inter_dim) = w2.shape
        blk_n = self.scale_block_n
        blk_k = self.scale_block_k
        h = hidden_states.float()
        h = h.view(B, -1, blk_k) * a_scale.unsqueeze(-1)
        h = h.view(B, -1)
        nblk_n = inter_dim // blk_n
        nblk_k = model_dim // blk_k
        fc1_s = fc1_scale.view(-1, 1).repeat(1, blk_n * blk_k).view(expert, -1, nblk_k, blk_n, blk_k)
        fc1_s = fc1_s.permute(0, 1, 3, 2, 4).reshape(expert, -1, model_dim)
        w1_dq = w1.float() * fc1_s
        fc2_s = fc2_scale.view(-1, 1).repeat(1, blk_k * blk_n).view(expert, nblk_k, nblk_n, blk_k, blk_n)
        fc2_s = fc2_s.permute(0, 1, 3, 2, 4).reshape(expert, model_dim, inter_dim)
        w2_dq = w2.float() * fc2_s
        h = h.view(B, 1, model_dim).repeat(1, top_k, 1)
        out = torch.zeros(B, top_k, D, dtype=torch.float32, device=hidden_states.device)
        for eid in range(expert):
            mask = topk_ids == eid
            if not mask.any():
                continue
            tokens = h[mask]
            act_input = tokens @ w1_dq[eid].T
            (gate, up) = act_input.split([inter_dim, inter_dim], dim=-1)
            act_out = F.silu(gate) * up
            out[mask] = act_out @ w2_dq[eid].T
        return (out * topk_weights.view(B, -1, 1)).sum(dim=1).to(hidden_states.dtype)
