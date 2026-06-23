import torch
import torch.nn as nn

class Model(nn.Module):
    _M_CHUNK = 16 * 1024

    def __init__(self, group_n: int, group_k: int) -> None:
        super().__init__()
        self.group_n = group_n
        self.group_k = group_k

    def forward(self, a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor) -> torch.Tensor:
        (m, k) = a.shape
        n = b.shape[0]
        num_k_blocks = k // self.group_k
        b_blocks = b.float().view(n, num_k_blocks, self.group_k).permute(1, 0, 2).contiguous()
        b_scale_per_n = b_scales.repeat_interleave(self.group_n, dim=0).t().contiguous()
        result = torch.empty(m, n, dtype=torch.float32, device=a.device)
        chunk = min(m, self._M_CHUNK) if m > 0 else 1
        for mc_start in range(0, m, chunk):
            mc_end = min(mc_start + chunk, m)
            cm = mc_end - mc_start
            a_blocks = a[mc_start:mc_end].float().view(cm, num_k_blocks, self.group_k).permute(1, 0, 2).contiguous()
            dots = torch.bmm(a_blocks, b_blocks.transpose(-2, -1))
            a_scale_chunk = a_scales[mc_start:mc_end].t().contiguous()
            scaled = dots * a_scale_chunk.unsqueeze(-1) * b_scale_per_n.unsqueeze(1)
            result[mc_start:mc_end] = scaled.sum(dim=0)
        return result
