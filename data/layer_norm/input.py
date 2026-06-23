import torch

def _make_inputs(shape: list[int]) -> dict[str, torch.Tensor]:
    tensor_shape = tuple((int(dim) for dim in shape))
    if len(tensor_shape) == 0:
        tensor_shape = (1,)
    x = torch.randn(tensor_shape, dtype=torch.bfloat16, device='cuda')
    feature_dim = tensor_shape[-1]
    weight = torch.randn(feature_dim, dtype=torch.bfloat16, device='cuda')
    bias = torch.randn(feature_dim, dtype=torch.bfloat16, device='cuda')
    return {'x': x, 'weight': weight, 'bias': bias}
