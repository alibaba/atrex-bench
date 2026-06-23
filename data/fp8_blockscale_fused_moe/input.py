import torch

def _make_inputs(token_count: int, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int, scale_block_n: int=128, scale_block_k: int=128) -> dict[str, torch.Tensor]:
    dt = torch.float8_e4m3fnuz
    fp8_max = torch.finfo(dt).max
    blk_n = scale_block_n
    blk_k = scale_block_k
    nblk_h_k = hidden_size // blk_k
    nblk_i_n = intermediate_size // blk_n
    nblk_2i_n = (intermediate_size * 2) // blk_n

    def _block_quant(x_fp32: torch.Tensor, group_dims: tuple) -> tuple:
        amax = x_fp32.abs().amax(dim=group_dims, keepdim=True).clamp(min=1e-12)
        scale = amax / fp8_max
        q = (x_fp32 / scale).clamp(-fp8_max, fp8_max).to(dt)
        return q, scale

    hidden_bf = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.03
    h_blocks = hidden_bf.float().view(token_count, nblk_h_k, blk_k)
    h_q, h_scale = _block_quant(h_blocks, group_dims=(-1,))
    hidden_states = h_q.view(token_count, hidden_size)
    a_scale = h_scale.squeeze(-1).to(torch.float32)

    w1_bf = torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=torch.bfloat16, device='cuda')
    w1_blocks = w1_bf.float().view(num_experts, nblk_2i_n, blk_n, nblk_h_k, blk_k)
    w1_q, w1_scale = _block_quant(w1_blocks, group_dims=(2, 4))
    w1 = w1_q.view(num_experts, intermediate_size * 2, hidden_size)
    fc1_scale = w1_scale.squeeze(2).squeeze(-1).to(torch.float32)

    w2_bf = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device='cuda')
    w2_blocks = w2_bf.float().view(num_experts, nblk_h_k, blk_k, nblk_i_n, blk_n)
    w2_q, w2_scale = _block_quant(w2_blocks, group_dims=(2, 4))
    w2 = w2_q.view(num_experts, hidden_size, intermediate_size)
    fc2_scale = w2_scale.squeeze(2).squeeze(-1).to(torch.float32)

    topk_ids = torch.topk(torch.rand(token_count, num_experts, device='cuda'), top_k, dim=1).indices.to(torch.int32)
    topk_weights = torch.softmax(torch.randn(token_count, top_k, device='cuda'), dim=-1).to(torch.float32)

    return {'hidden_states': hidden_states, 'w1': w1, 'w2': w2, 'topk_weights': topk_weights, 'topk_ids': topk_ids, 'a_scale': a_scale, 'fc1_scale': fc1_scale, 'fc2_scale': fc2_scale}

def get_inputs() -> list[torch.Tensor]:
    return list(_make_inputs(token_count=32, hidden_size=2048, intermediate_size=768, num_experts=128, top_k=8).values())

def get_init_inputs() -> dict[str, object]:
    return {'num_experts': 128, 'intermediate_size': 768, 'top_k': 8}
