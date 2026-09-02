# models/upsample.py
"""
EMA Dechunking (Upsampler) — paper Eq. (5).

FIX: The original used torch.empty_like() + h[t] = ... (in-place writes).
This corrupts the autograd graph because the pre-allocated tensor is a
tracked variable and in-place mutation at version t conflicts with the
backward pass needing version t-1.

Fix: accumulate steps in a plain Python list, then torch.stack at the end.
No in-place writes → no autograd version conflicts.
"""

import torch
import torch.nn as nn
from typing import List


class EMADeChunk(nn.Module):
    """
    Upsample chunk-level representations back to byte resolution
    using an EMA governed by boundary probabilities.

    z̄_t = p_t * ẑ_{chunk(t)} + (1 - p_t) * z̄_{t-1}
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        chunk_embs_batch: List[torch.Tensor],  # List of [C_i, D]
        chunk_ids_batch:  List[torch.Tensor],  # List of [T_i]
        p:                torch.Tensor,         # [B, T]
        attention_mask:   torch.Tensor,         # [B, T]
    ) -> torch.Tensor:
        """Returns h_hat: [B, T, D]. Padded positions filled with zeros."""
        B = len(chunk_embs_batch)
        D = chunk_embs_batch[0].shape[-1]
        T = p.shape[1]

        outputs = []

        for i in range(B):
            z    = chunk_embs_batch[i]              # [C_i, D]
            cids = chunk_ids_batch[i]               # [T_i]
            p_i  = p[i]                             # [T]
            mask = attention_mask[i]                # [T]

            T_i = int(mask.sum().item())

            # Expand chunk embeddings to byte resolution via index lookup
            cids_safe  = cids[:T_i].clamp(0, z.shape[0] - 1)
            z_expanded = z[cids_safe]               # [T_i, D]

            # EMA scan (no in-place writes)
            p_i_valid = p_i[:T_i].to(dtype=z.dtype)
            h_out = _ema_scan(z_expanded, p_i_valid)   # [T_i, D]

            # Pad back to full sequence length T
            if T_i < T:
                pad   = torch.zeros(T - T_i, D, device=z.device, dtype=z.dtype)
                h_out = torch.cat([h_out, pad], dim=0)

            outputs.append(h_out)

        return torch.stack(outputs, dim=0)   # [B, T, D]


def _ema_scan(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Sequential EMA scan — autograd-safe.

      z : [T, D]  expanded chunk embeddings
      p : [T]     boundary probabilities

    h[0] = z[0]
    h[t] = p[t] * z[t] + (1 - p[t]) * h[t-1]

    KEY FIX: accumulate into a Python list and torch.stack at the end.
    Never write in-place into a pre-allocated tensor while autograd is active.
    """
    p_col = p.unsqueeze(1)          # [T, 1]

    steps = [z[0]]                  # list of [D] tensors — no in-place writes

    for t in range(1, z.shape[0]):
        steps.append(p_col[t] * z[t] + (1.0 - p_col[t]) * steps[-1])

    return torch.stack(steps, dim=0)   # [T, D]