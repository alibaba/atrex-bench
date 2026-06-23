import torch

def _make_inputs(seq_lens: list[int], num_heads: int, head_dim: int) -> dict[str, torch.Tensor]:
    total_tokens = sum(seq_lens)
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    cu_seqlens_q = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device='cuda')
    for i, sl in enumerate(seq_lens):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + sl
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + sl
    return {'q': q, 'k': k, 'v': v, 'cu_seqlens_q': cu_seqlens_q, 'cu_seqlens_k': cu_seqlens_k}
