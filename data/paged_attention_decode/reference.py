import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_tables: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        (num_seqs, num_query_heads, head_size) = query.shape
        num_kv_heads = v_cache.shape[1]
        block_size = v_cache.shape[3]
        num_queries_per_kv = num_query_heads // num_kv_heads
        scale = head_size ** (-0.5)
        k_cache_flat = k_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_kv_heads, head_size)
        v_cache_flat = v_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_kv_heads, head_size)
        seq_lens_list = seq_lens.tolist()
        block_tables_list = block_tables.tolist()
        output = torch.zeros_like(query)
        for seq_idx in range(num_seqs):
            ctx_len = int(seq_lens_list[seq_idx])
            if ctx_len <= 0:
                continue
            blocks = block_tables_list[seq_idx]
            blocks_t = torch.as_tensor(blocks[:(ctx_len + block_size - 1) // block_size], dtype=torch.long, device=query.device)
            within = torch.arange(ctx_len, device=query.device, dtype=torch.long)
            idx = blocks_t[within // block_size] * block_size + within % block_size
            keys = k_cache_flat.index_select(0, idx).float()
            values = v_cache_flat.index_select(0, idx).float()
            if num_queries_per_kv > 1:
                keys = keys.repeat_interleave(num_queries_per_kv, dim=1)
                values = values.repeat_interleave(num_queries_per_kv, dim=1)
            q_seq = query[seq_idx].float().unsqueeze(0).unsqueeze(2)
            ks = keys.transpose(0, 1).unsqueeze(0)
            vs = values.transpose(0, 1).unsqueeze(0)
            os = F.scaled_dot_product_attention(q_seq, ks, vs, is_causal=False)
            output[seq_idx] = os.squeeze(0).squeeze(1).to(query.dtype)
        return output
