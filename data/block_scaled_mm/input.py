import torch
_M_CHUNK = 64 * 1024

def _make_inputs(m: int, n: int, k: int, group_n: int, group_k: int) -> dict[str, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    b_raw = torch.randn(n, k, dtype=torch.float32, device='cuda')
    b_max = b_raw.view(n // group_n, group_n, k // group_k, group_k).amax(dim=1).amax(dim=-1)
    b_scales = torch.clamp(b_max / fp8_max, min=1e-10)
    b_quant = (b_raw / b_scales.repeat_interleave(group_n, dim=0).repeat_interleave(group_k, dim=1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    del b_raw
    a_quant = torch.empty(m, k, dtype=torch.float8_e4m3fn, device='cuda')
    a_scales = torch.empty(m, k // group_k, dtype=torch.float32, device='cuda')
    chunk = min(m, _M_CHUNK) if m > 0 else 1
    for mc_start in range(0, m, chunk):
        mc_end = min(mc_start + chunk, m)
        a_raw_c = torch.randn(mc_end - mc_start, k, dtype=torch.float32, device='cuda')
        a_max_c = a_raw_c.abs().view(mc_end - mc_start, k // group_k, group_k).amax(dim=-1)
        a_scales_c = torch.clamp(a_max_c / fp8_max, min=1e-10)
        a_quant_c = (a_raw_c / a_scales_c.repeat_interleave(group_k, dim=1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        a_quant[mc_start:mc_end] = a_quant_c
        a_scales[mc_start:mc_end] = a_scales_c
    return {'a': a_quant, 'b': b_quant, 'a_scales': a_scales, 'b_scales': b_scales}
