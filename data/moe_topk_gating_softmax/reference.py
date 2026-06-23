import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, top_k: int, num_expert_groups: int = 0, topk_group: int = 0) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_expert_groups = num_expert_groups
        self.topk_group = topk_group

    def forward(self, gating_output: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = torch.softmax(gating_output.float(), dim=-1)
        if self.num_expert_groups > 0 and self.topk_group > 0:
            n, ne = scores.shape
            group_size = ne // self.num_expert_groups
            grouped = scores.view(n, self.num_expert_groups, group_size)
            group_scores = grouped.amax(dim=-1)
            _, top_groups = torch.topk(group_scores, k=self.topk_group, dim=-1)
            mask = torch.zeros(n, self.num_expert_groups, dtype=torch.bool, device=scores.device)
            mask.scatter_(1, top_groups, True)
            mask = mask.unsqueeze(-1).expand(-1, -1, group_size).reshape(n, ne)
            scores = scores.masked_fill(~mask, 0.0)
        (topk_weights, topk_ids) = torch.topk(scores, k=self.top_k, dim=-1)
        return {'topk_weights': topk_weights, 'topk_ids': topk_ids.to(torch.int32)}
