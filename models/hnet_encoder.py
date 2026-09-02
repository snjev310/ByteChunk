# models/hnet_encoder.py
import torch
import torch.nn as nn
from typing import Tuple, Dict
from mamba_ssm import Mamba

from models.routing import RoutingModule, RoutingOutput
from models.chunking import DynamicChunking, pad_chunks
from models.smoothing import SmoothingModule
from models.upsample import EMADeChunk


class HNetEncoder(nn.Module):
    def __init__(
        self,
        byte_embedding:  nn.Embedding,
        routing:         RoutingModule,
        chunker:         DynamicChunking,
        smoother:        SmoothingModule,
        ema_dechunk:     EMADeChunk,
        llama:           nn.Module,
        d_model:         int,
        boundary_thresh: float = 0.3,
        n_local_layers:  int   = 1,
    ):
        super().__init__()
        self.byte_embedding  = byte_embedding
        self.routing         = routing
        self.chunker         = chunker
        self.smoother        = smoother
        self.ema_dechunk     = ema_dechunk
        self.llama           = llama
        self.boundary_thresh = boundary_thresh

        self.skip_proj = nn.Linear(d_model, d_model, bias=False)

        # N independent Mamba layers (not shared weights)
        self.local_encoder = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_local_layers)
        ])
        self.local_decoder = nn.ModuleList([
            Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2)
            for _ in range(n_local_layers)
        ])

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, T]
        attention_mask: torch.Tensor,   # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        B, T = input_ids.shape

        # ── 1. Byte Embedding 
        h = self.byte_embedding(input_ids)
        h = h.to(dtype=self.routing.W_q.weight.dtype)

        # ── 2. Local Encoder (N independent Mamba layers) 
        for layer in self.local_encoder:
            h = h + layer(h)

        # ── 3. Routing 
        router_out: RoutingOutput = self.routing(h)

        pos0_mask = torch.ones(B, T, device=input_ids.device, dtype=router_out.p.dtype)
        pos0_mask[:, 0] = 0.0
        p = router_out.p * attention_mask.float() * pos0_mask

        # ── 4. Hard boundaries 
        b_hard = (p >= self.boundary_thresh).float()

        # ── 5. Dynamic Chunking ───────────────────────────────────────────
        chunk_out  = self.chunker(h, b_hard, attention_mask)
        chunk_embs = chunk_out["chunk_embs"]     # List[Tensor[C_i, D]]
        chunk_ids  = chunk_out["chunk_ids"]      # List[Tensor[T_i]]

        # ── 6. Smoothing ──────────────────────────────────────────────────
        chunk_embs_smooth = [self.smoother(c) for c in chunk_embs]

        # Store smoothed embs in chunk_out for alignment loss access
        chunk_out["chunk_embs_smooth"] = chunk_embs_smooth

        # ── 7. LLaMA ──────────────────────────────────────────────────────
        padded_chunks, chunk_mask = pad_chunks(chunk_embs_smooth)

        llama_out = self.llama(
            inputs_embeds  = padded_chunks,
            attention_mask = chunk_mask,
            use_cache      = False,
            return_dict    = True,
        )
        z_chunk = llama_out.last_hidden_state    # [B, C_max, D]

        z_chunk_list = [
            z_chunk[i, :chunk_embs_smooth[i].shape[0], :]
            for i in range(B)
        ]

        # ── 8. EMA Dechunking ─────────────────────────────────────────────
        h_dechunk = self.ema_dechunk(
            chunk_embs_batch = z_chunk_list,
            chunk_ids_batch  = chunk_ids,
            p                = p,
            attention_mask   = attention_mask,
        )

        # ── 9. Skip connection ────────────────────────────────────────────
        skip_byte = _expand_chunks_to_bytes(chunk_embs_smooth, chunk_ids, attention_mask)
        h_hat     = h_dechunk + self.skip_proj(skip_byte)

        # ── 10. Local Decoder (N independent Mamba layers) ────────────────
        for layer in self.local_decoder:
            h_hat = h_hat + layer(h_hat)

        aux = {
            "chunk_out":  chunk_out,
            "b_hard":     b_hard,
            "router_out": router_out,
            "num_chunks": [c.shape[0] for c in chunk_embs_smooth],
        }

        return h_hat, p, aux


def _expand_chunks_to_bytes(
    chunk_embs_batch,
    chunk_ids_batch,
    attention_mask,
) -> torch.Tensor:
    B = len(chunk_embs_batch)
    T = attention_mask.shape[1]
    D = chunk_embs_batch[0].shape[-1]
    device = chunk_embs_batch[0].device
    dtype  = chunk_embs_batch[0].dtype

    samples = []
    for i in range(B):
        c     = chunk_embs_batch[i]
        cids  = chunk_ids_batch[i]
        T_i   = int(attention_mask[i].sum().item())

        cids_safe = cids[:T_i].clamp(0, c.shape[0] - 1)
        gathered  = c[cids_safe]

        if T_i < T:
            pad      = torch.zeros(T - T_i, D, device=device, dtype=dtype)
            gathered = torch.cat([gathered, pad], dim=0)

        samples.append(gathered)

    return torch.stack(samples, dim=0)