import torch

def _make_inputs(num_seqs: int, seq_lens: list[int], num_query_heads: int, num_kv_heads: int, head_size: int, block_size: int) -> dict[str, torch.Tensor]:
    num_tokens = sum(seq_lens)
    max_seq_len = max(seq_lens)
    max_num_blocks = (max_seq_len + block_size - 1) // block_size + 2
    q = torch.randn(num_tokens, num_query_heads, head_size, dtype=torch.bfloat16, device='cuda')
    num_blocks = max_num_blocks * num_seqs + 64
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16, device='cuda')
    cu_seqlens_q = torch.tensor([0] + [sum(seq_lens[:index + 1]) for index in range(num_seqs)], dtype=torch.int32, device='cuda')
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device='cuda')
    block_table = torch.zeros(num_seqs, max_num_blocks, dtype=torch.int32, device='cuda')
    for seq_index in range(num_seqs):
        num_blocks_needed = (seq_lens[seq_index] + block_size - 1) // block_size
        block_table[seq_index, :num_blocks_needed] = torch.arange(seq_index * max_num_blocks, seq_index * max_num_blocks + num_blocks_needed, dtype=torch.int32, device='cuda')
    return {'q': q, 'k_cache': k_cache, 'v_cache': v_cache, 'cu_seqlens_q': cu_seqlens_q, 'seqused_k': seqused_k, 'block_table': block_table}
