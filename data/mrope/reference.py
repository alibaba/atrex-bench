import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, head_size: int, rotary_dim: int, mrope_section: list[int], mrope_interleaved: bool=False, is_neox_style: bool=True) -> None:
        super().__init__()
        self.head_size = int(head_size)
        self.rotary_dim = int(rotary_dim)
        self.mrope_section = [int(x) for x in mrope_section]
        self.mrope_interleaved = bool(mrope_interleaved)
        self.is_neox_style = bool(is_neox_style)

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> dict[str, torch.Tensor]:
        return {'q': self._rotate(q, cos, sin), 'k': self._rotate(k, cos, sin)}

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        heads = x.shape[1] // self.head_size
        x_view = x.reshape(tokens, heads, self.head_size).clone()
        half = self.rotary_dim // 2
        axis = self._axis_map(half, x.device)
        pos = torch.arange(tokens, device=x.device)[:, None]
        freq = torch.arange(half, device=x.device)[None, :]
        cos_row = cos[axis[None, :], pos, freq]
        sin_row = sin[axis[None, :], pos, freq]
        if self.is_neox_style:
            x1 = x_view[:, :, :half].float()
            x2 = x_view[:, :, half:self.rotary_dim].float()
            c = cos_row[:, None, :].float()
            s = sin_row[:, None, :].float()
            x_view[:, :, :half] = (x1 * c - x2 * s).to(x.dtype)
            x_view[:, :, half:self.rotary_dim] = (x2 * c + x1 * s).to(x.dtype)
        else:
            even = x_view[:, :, :self.rotary_dim:2].float()
            odd = x_view[:, :, 1:self.rotary_dim:2].float()
            c = cos_row[:, None, :].float()
            s = sin_row[:, None, :].float()
            x_view[:, :, :self.rotary_dim:2] = (even * c - odd * s).to(x.dtype)
            x_view[:, :, 1:self.rotary_dim:2] = (odd * c + even * s).to(x.dtype)
        return x_view.reshape_as(x)

    def _axis_map(self, half: int, device: torch.device) -> torch.Tensor:
        offsets = torch.arange(half, device=device)
        if self.mrope_interleaved:
            h = (offsets % 3 == 1) & (offsets <= 3 * self.mrope_section[1])
            w = (offsets % 3 == 2) & (offsets <= 3 * self.mrope_section[2])
            return torch.where(h, 1, torch.where(w, 2, 0)).long()
        t_end = self.mrope_section[0]
        h_end = t_end + self.mrope_section[1]
        return torch.where(offsets < t_end, 0, torch.where(offsets < h_end, 1, 2)).long()
