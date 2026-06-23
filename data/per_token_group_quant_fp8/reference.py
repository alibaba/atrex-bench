import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (m, n) = x.shape
        if getattr(torch.version, 'hip', None) and hasattr(torch, 'float8_e4m3fnuz'):
            fp8_dtype = torch.float8_e4m3fnuz
            fp8_max = 224.0
        else:
            fp8_dtype = torch.float8_e4m3fn
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
        x_fp32 = x.float()
        x_groups = x_fp32.view(m, n // self.group_size, self.group_size)
        x_abs_max = x_groups.abs().amax(dim=-1)
        scales = torch.clamp(x_abs_max / fp8_max, min=1e-10)
        quantized = (x_fp32 / scales.repeat_interleave(self.group_size, dim=1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return {'quantized': quantized, 'scales': scales}
