import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, nhead: int, kv_lora_rank: int, qk_rope_head_dim: int) -> None:
        super().__init__()
        self.nhead = nhead
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = kv_lora_rank + qk_rope_head_dim
        self.v_head_dim = kv_lora_rank

    def forward(self, q: torch.Tensor, kv_cache: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        batch_size = seq_lens.shape[0]
        out = torch.empty(batch_size, self.nhead, self.v_head_dim, dtype=torch.float32, device=q.device)
        seq_lens_list = seq_lens.tolist()
        kv_offset = 0
        for i in range(batch_size):
            seq_len = int(seq_lens_list[i])
            kvc_i = kv_cache[kv_offset:kv_offset + seq_len].float()
            q_i = q[i].float().unsqueeze(0).unsqueeze(2)
            k_i = kvc_i.permute(1, 0, 2).expand(self.nhead, seq_len, self.qk_head_dim).unsqueeze(0)
            v_i = kvc_i[..., :self.kv_lora_rank].permute(1, 0, 2).expand(self.nhead, seq_len, self.v_head_dim).unsqueeze(0)
            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False)
            out[i] = o_i.squeeze(0).squeeze(1)
            kv_offset += seq_len
        return out.to(q.dtype)
