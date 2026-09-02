# models/smoothing.py
"""
Smoothing Module — paper §2.2.2.

Applied to chunk-level embeddings BEFORE they enter the main network.
Converts discrete chunk representations into smoother interpolated ones
using a simple MLP residual block with LayerNorm.

Input/output shape: [C, D]  (works on a single sequence's chunks)
or batched:         [B, C, D]
"""

import torch
import torch.nn as nn


class SmoothingModule(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_model * expansion, bias=False),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model, bias=False),
        )

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        """
        chunks: [..., D]   any leading batch dims are fine
        returns: [..., D]  same shape, residual-refined
        """
        return chunks + self.mlp(self.norm(chunks))
