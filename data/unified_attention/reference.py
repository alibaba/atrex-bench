import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cu_seqlens_q: torch.Tensor, seqused_k: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        (num_tokens, num_query_heads, head_size) = q.shape
        num_kv_heads = k_cache.shape[2]
        block_size = k_cache.shape[1]
        num_queries_per_kv = num_query_heads // num_kv_heads
        num_seqs = seqused_k.numel()
        device = q.device
        k_flat = k_cache.contiguous().view(-1, num_kv_heads, head_size)
        v_flat = v_cache.contiguous().view(-1, num_kv_heads, head_size)
        q_bounds = cu_seqlens_q.tolist()
        k_lens = seqused_k.tolist()
        out = torch.empty(num_tokens, num_query_heads, head_size, dtype=q.dtype, device=device)
        for seq_idx in range(num_seqs):
            q_start, q_end = q_bounds[seq_idx], q_bounds[seq_idx + 1]
            q_len = q_end - q_start
            kv_len = int(k_lens[seq_idx])
            if q_len == 0:
                continue
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            flat_indices = []
            for b in range(num_kv_blocks):
                block_id = int(block_table[seq_idx, b].item())
                remaining = min(block_size, kv_len - b * block_size)
                for t in range(remaining):
                    flat_indices.append(block_id * block_size + t)
            flat_idx = torch.tensor(flat_indices, dtype=torch.long, device=device)
            k_seq = k_flat.index_select(0, flat_idx)
            v_seq = v_flat.index_select(0, flat_idx)
            qs = q[q_start:q_end].transpose(0, 1).unsqueeze(0).float()
            ks = k_seq.transpose(0, 1).unsqueeze(0).float()
            vs = v_seq.transpose(0, 1).unsqueeze(0).float()
            if num_queries_per_kv > 1:
                ks = ks.repeat_interleave(num_queries_per_kv, dim=1)
                vs = vs.repeat_interleave(num_queries_per_kv, dim=1)
            is_causal = (q_len == kv_len)
            os = F.scaled_dot_product_attention(qs, ks, vs, is_causal=is_causal)
            out[q_start:q_end] = os.squeeze(0).transpose(0, 1).to(q.dtype)
        return out
