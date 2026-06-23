import torch


def _make_inputs(rows: int, cols: int) -> dict[str, torch.Tensor]:
    return {"x": torch.randn(rows, cols)}
