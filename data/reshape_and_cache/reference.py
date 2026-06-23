import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, slot_mapping: torch.Tensor) -> dict[str, torch.Tensor]:
        key_out = key_cache.clone()
        value_out = value_cache.clone()
        valid = slot_mapping >= 0
        slots = slot_mapping[valid].long()
        key_out.view(-1, key.shape[1], key.shape[2])[slots] = key[valid].to(key_out.dtype)
        value_out.view(-1, value.shape[1], value.shape[2])[slots] = value[valid].to(value_out.dtype)
        return {'key_cache': key_out, 'value_cache': value_out}
