"""
Byte-level LM Head.
Projects hidden states [B, T, D] → logits [B, T, 256].
Optional weight tying with byte embedding.
"""

import torch
import torch.nn as nn


class ByteLMHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int = 256, weight=None):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        if weight is not None:
            if weight.shape != self.proj.weight.shape:
                raise ValueError(
                    f"Weight tying shape mismatch: "
                    f"got {weight.shape}, expected {self.proj.weight.shape}"
                )
            self.proj.weight = weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, D]  →  [B, T, V]"""
        return self.proj(h)
