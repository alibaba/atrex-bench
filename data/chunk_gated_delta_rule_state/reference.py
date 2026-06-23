import math
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, chunk_size: int=64) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)

    def forward(self, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor, initial_state_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        (batch, tokens, num_k_heads, key_dim) = k.shape
        (num_heads, value_dim) = (u.shape[2], u.shape[3])
        heads_per_k = max(num_heads // num_k_heads, 1)
        cs = self.chunk_size
        num_chunks = math.ceil(tokens / cs)
        head_to_khead = torch.tensor([min(hd // heads_per_k, num_k_heads - 1) for hd in range(num_heads)], dtype=torch.long, device=k.device)
        k_h = k.index_select(2, head_to_khead)
        final_state = initial_state.float().clone()
        idx_list = initial_state_indices.tolist()
        state_indices = torch.tensor(idx_list, dtype=torch.long, device=k.device)
        h = final_state.index_select(0, state_indices).clone()
        h_chunks = torch.empty(batch, num_chunks, num_heads, key_dim, value_dim, dtype=k.dtype, device=k.device)
        v_new = torch.empty_like(u)
        BH = batch * num_heads
        h_flat = h.reshape(BH, key_dim, value_dim)
        for chunk_idx in range(num_chunks):
            start = chunk_idx * cs
            end = min(start + cs, tokens)
            cur_cs = end - start
            h_chunks[:, chunk_idx] = h_flat.reshape(batch, num_heads, key_dim, value_dim).to(k.dtype)
            w_blk = w[:, start:end].float()
            u_blk = u[:, start:end].float()
            k_blk = k_h[:, start:end].float()
            g_blk = g[:, start:end].float()
            w_bh = w_blk.permute(0, 2, 1, 3).reshape(BH, cur_cs, key_dim)
            u_bh = u_blk.permute(0, 2, 1, 3).reshape(BH, cur_cs, value_dim)
            k_bh = k_blk.permute(0, 2, 1, 3).reshape(BH, cur_cs, key_dim)
            g_bh = g_blk.permute(0, 2, 1).reshape(BH, cur_cs)
            v_blk = u_bh - torch.bmm(w_bh.to(k.dtype).float(), h_flat.to(k.dtype).float())
            v_blk_out = v_blk.reshape(batch, num_heads, cur_cs, value_dim).permute(0, 2, 1, 3)
            v_new[:, start:end] = v_blk_out.to(u.dtype)
            g_last = g_bh[:, -1:]
            g_diff = g_last - g_bh
            v_blk_scaled = v_blk * torch.exp(torch.where(g_diff <= 0, g_diff, torch.tensor(float('-inf'), device=g_diff.device))).unsqueeze(-1)
            v_blk_scaled = v_blk_scaled.to(k.dtype)
            h_flat = h_flat * torch.exp(g_last).unsqueeze(-1)
            h_flat = h_flat + torch.bmm(k_bh.to(k.dtype).transpose(-2, -1).float(), v_blk_scaled.float())
        h_back = h_flat.reshape(batch, num_heads, key_dim, value_dim)
        final_state.index_copy_(0, state_indices, h_back)
        return {'h': h_chunks, 'v_new': v_new, 'final_state': final_state}
