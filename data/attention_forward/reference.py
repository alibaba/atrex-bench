import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(q)
        q_bounds = cu_seqlens_q.tolist()
        k_bounds = cu_seqlens_k.tolist()
        for seq_idx in range(len(q_bounds) - 1):
            qs_start, qs_end = q_bounds[seq_idx], q_bounds[seq_idx + 1]
            ks_start, ks_end = k_bounds[seq_idx], k_bounds[seq_idx + 1]
            if qs_end <= qs_start:
                continue
            qs = q[qs_start:qs_end].transpose(0, 1).unsqueeze(0).float()
            ks = k[ks_start:ks_end].transpose(0, 1).unsqueeze(0).float()
            vs = v[ks_start:ks_end].transpose(0, 1).unsqueeze(0).float()
            os = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False)
            out[qs_start:qs_end] = os.squeeze(0).transpose(0, 1).to(q.dtype)
        return out
