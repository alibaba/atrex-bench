"""Candidate Model for atrex_001 (ReLU).

This is a pure-PyTorch wrapper used for testing the eval pipeline
without requiring Triton/GPU. In production, this would be a Triton kernel.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Triton-equivalent ReLU (pure PyTorch for testing)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
