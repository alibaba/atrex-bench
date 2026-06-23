import torch


def _make_inputs(batch_size: int, ctx_len: int, nhead: int, kv_lora_rank: int, qk_rope_head_dim: int) -> dict[str, torch.Tensor]:
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    q = torch.randn(batch_size, nhead, qk_head_dim, dtype=torch.bfloat16, device='cuda')
    total_kv = batch_size * ctx_len
    kv_cache = torch.randn(total_kv, 1, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16, device='cuda')
    seq_lens = torch.full((batch_size,), ctx_len, dtype=torch.int32, device='cuda')
    return {'q': q, 'kv_cache': kv_cache, 'seq_lens': seq_lens}
