import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, softplus_beta: float=1.0, softplus_threshold: float=20.0, scale: float | None=None, use_qk_l2norm: bool=True) -> None:
        super().__init__()
        self.softplus_beta = float(softplus_beta)
        self.softplus_threshold = float(softplus_threshold)
        self.scale = scale
        self.use_qk_l2norm = bool(use_qk_l2norm)

    def forward(self, A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor, initial_state: torch.Tensor, initial_state_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        (batch, tokens, num_q_heads, key_dim) = q.shape
        (num_v_heads, value_dim) = (v.shape[2], v.shape[3])
        heads_per_q = max(num_v_heads // num_q_heads, 1)
        scale = self.scale if self.scale is not None else key_dim ** (-0.5)
        hv_to_hq = torch.tensor([min(hv // heads_per_q, num_q_heads - 1) for hv in range(num_v_heads)], dtype=torch.long, device=q.device)
        state = initial_state.float().clone()
        idx_list = initial_state_indices.tolist()
        state_indices = torch.tensor(idx_list, dtype=torch.long, device=q.device)
        h = state.index_select(0, state_indices).clone()
        BH = batch * num_v_heads
        h_flat = h.reshape(BH, key_dim, value_dim)
        A_log_v = A_log.float()
        dt_bias_v = dt_bias.float()
        A_log_bh = A_log_v.unsqueeze(0).expand(batch, num_v_heads).reshape(BH)
        dt_bias_bh = dt_bias_v.unsqueeze(0).expand(batch, num_v_heads).reshape(BH)
        q_h = q.index_select(2, hv_to_hq).float()
        k_h = k.index_select(2, hv_to_hq).float()
        v_h = v.float()
        a_h = a.float()
        b_h = b.float()
        gate_all = F.softplus(a_h + dt_bias_v.unsqueeze(0).unsqueeze(0), beta=self.softplus_beta, threshold=self.softplus_threshold)
        gate_all = gate_all * (-torch.exp(A_log_v).unsqueeze(0).unsqueeze(0))
        beta_all = torch.sigmoid(b_h)
        out = torch.empty_like(v)
        for t in range(tokens):
            q_t = q_h[:, t].reshape(BH, key_dim)
            k_t = k_h[:, t].reshape(BH, key_dim)
            v_t = v_h[:, t].reshape(BH, value_dim)
            if self.use_qk_l2norm:
                q_t = F.normalize(q_t, p=2.0, dim=-1, eps=1e-06)
                k_t = F.normalize(k_t, p=2.0, dim=-1, eps=1e-06)
            g = gate_all[:, t].reshape(BH)
            beta = beta_all[:, t].reshape(BH)
            h_flat = h_flat * torch.exp(g).view(BH, 1, 1)
            hk = (h_flat * k_t.unsqueeze(-1)).sum(dim=1)
            v_new = (v_t - hk) * beta.unsqueeze(-1)
            h_flat = h_flat + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
            q_scaled = q_t * scale
            out_t = (h_flat * q_scaled.unsqueeze(-1)).sum(dim=1)
            out[:, t] = out_t.reshape(batch, num_v_heads, value_dim).to(v.dtype)
        final_h = h_flat.reshape(batch, num_v_heads, key_dim, value_dim)
        state.index_copy_(0, state_indices, final_h)
        return {'out': out, 'final_state': state.to(initial_state.dtype)}
