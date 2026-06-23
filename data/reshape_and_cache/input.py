import torch

def _make_inputs(shape: list[int], block_size: int=16) -> dict[str, torch.Tensor]:
    (num_tokens, num_heads, head_size) = (int(dim) for dim in shape)
    block_size = int(block_size)
    num_blocks = (num_tokens + block_size - 1) // block_size
    key = torch.randn((num_tokens, num_heads, head_size), dtype=torch.bfloat16, device='cuda')
    value = torch.randn((num_tokens, num_heads, head_size), dtype=torch.bfloat16, device='cuda')
    key_cache = torch.zeros((num_blocks, block_size, num_heads, head_size), dtype=torch.bfloat16, device='cuda')
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device='cuda')
    return {'key': key, 'value': value, 'key_cache': key_cache, 'value_cache': value_cache, 'slot_mapping': slot_mapping}
