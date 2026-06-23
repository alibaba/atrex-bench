import torch

def _make_inputs(num_seqs: int, ctx_len: int, num_query_heads: int, num_kv_heads: int, head_size: int, block_size: int=16, dtype: str='bf16') -> dict[str, torch.Tensor]:
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    max_num_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs + 16
    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dt, device='cuda')
    x = 16 // dt.itemsize
    k_cache = torch.randn(num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=dt, device='cuda')
    v_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=dt, device='cuda')
    block_tables = torch.zeros(num_seqs, max_num_blocks_per_seq, dtype=torch.int32, device='cuda')
    for i in range(num_seqs):
        for j in range(max_num_blocks_per_seq):
            block_tables[i, j] = i * max_num_blocks_per_seq + j
    seq_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32, device='cuda')
    return {'query': query, 'k_cache': k_cache, 'v_cache': v_cache, 'block_tables': block_tables, 'seq_lens': seq_lens}
