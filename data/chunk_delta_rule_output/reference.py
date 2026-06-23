import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, chunk_size: int=64, scale: float | None=None) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        (batch, tokens, num_q_heads, key_dim) = q.shape
        (num_heads, value_dim) = (v.shape[2], v.shape[3])
        heads_per_q = max(num_heads // num_q_heads, 1)
        cs = self.chunk_size
        scale = self.scale if self.scale is not None else key_dim ** (-0.5)
        num_chunks = math.ceil(tokens / cs)
        tokens_padded = num_chunks * cs
        pad = tokens_padded - tokens
        head_to_qhead = torch.tensor([min(hd // heads_per_q, num_q_heads - 1) for hd in range(num_heads)], dtype=torch.long, device=q.device)
        q_h = q.index_select(2, head_to_qhead)
        k_h = k.index_select(2, head_to_qhead)
        if pad > 0:
            q_h = F.pad(q_h, (0, 0, 0, 0, 0, pad))
            k_h = F.pad(k_h, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            g = F.pad(g, (0, 0, 0, pad))

        def chunkify(x: torch.Tensor) -> torch.Tensor:
            (B, T, H, D) = x.shape
            return x.reshape(B, num_chunks, cs, H, D).permute(0, 3, 1, 2, 4).reshape(B * H * num_chunks, cs, D)
        q_blk = chunkify(q_h)
        k_blk = chunkify(k_h)
        v_blk = chunkify(v)
        g_blk = chunkify(g.unsqueeze(-1)).float().squeeze(-1)
        h_blk = h.permute(0, 2, 1, 3, 4).reshape(batch * num_heads * num_chunks, key_dim, value_dim)
        o = torch.bmm(q_blk.float(), h_blk.float())
        attn = torch.bmm(q_blk.float(), k_blk.float().transpose(-2, -1))
        exp_g = torch.exp(g_blk)
        o = o * exp_g.unsqueeze(-1)
        g_diff = g_blk.unsqueeze(-1) - g_blk.unsqueeze(-2)
        attn = attn * torch.exp(torch.where(g_diff <= 0, g_diff, torch.tensor(float('-inf'), device=g_diff.device)))
        attn = torch.tril(attn)
        o_full = (o + torch.bmm(attn.to(v.dtype).float(), v_blk.float())) * scale
        out = o_full.reshape(batch, num_heads, num_chunks, cs, value_dim).permute(0, 2, 3, 1, 4).reshape(batch, tokens_padded, num_heads, value_dim).to(v.dtype)
        return out[:, :tokens]
