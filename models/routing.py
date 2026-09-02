# models/routing.py
"""
Routing Module — paper-faithful implementation of Eq. (4).

Given byte embeddings h [B, T, D]:
  q_t = W_q * h_t
  k_t = W_k * h_t
  sim_t = cosine(q_t, k_{t-1})          # adjacent similarity
  p_t = 0.5 * (1 - sim_t)               # boundary probability
  b_t = 1{p_t >= 0.5}                   # hard boundary indicator

Convention:
  - p[:,0] = 0.0  (no boundary at first token — cumsum starts chunk 0)
  - b[:,0] = 0    (first token always belongs to chunk 0)

Note: paper says p_1 = 1.0 (1-indexed), but because chunking uses
cumsum(b), whether t=0 is a boundary or not doesn't create a new chunk —
chunk 0 always exists.  We zero p[:,0] for consistency with the EMA upsampler.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RoutingOutput:
    p:   torch.Tensor    # [B, T]  boundary probabilities ∈ [0, 1]
    sim: torch.Tensor    # [B, T]  cosine similarities
    q:   torch.Tensor    # [B, T, D]
    k:   torch.Tensor    # [B, T, D]


class RoutingModule(nn.Module):
    """
    Learnable boundary router.
    Projects byte embeddings and computes adjacent cosine similarity.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> RoutingOutput:
        """
        h: [B, T, D]  byte embeddings (any float dtype)
        """
        q = self.W_q(h)   # [B, T, D]
        k = self.W_k(h)   # [B, T, D]

        # fp32 for numerical stability of cosine sim
        q_f = q.float()
        k_f = k.float()

        # cosine similarity between adjacent positions: sim_t = cos(q_t, k_{t-1})
        # shape: [B, T-1]
        # sim = F.cosine_similarity(q_f[:, 1:, :], k_f[:, :-1, :], dim=-1)
        
        # Adding non casulatity to the routing module by swapping q and k in cosine similarity {BoLMO paper does this in their implementation}
        sim = F.cosine_similarity(k_f[:, 1:, :], q_f[:, :-1, :], dim=-1)

        # prepend sim_0 = 1.0  →  p_0 = 0.5*(1-1) = 0.0  (no boundary)
        sim = F.pad(sim, (1, 0), value=1.0)   # [B, T]

        # Eq. (4)
        p = 0.5 * (1.0 - sim)
        p = p.clamp(0.0, 1.0)

        return RoutingOutput(p=p, sim=sim, q=q_f, k=k_f)
