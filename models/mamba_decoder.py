# models/mamba_decoder.py
"""
Hybrid Mamba + Cross-Attention decoder for byte-level MT.

Each decoder layer:
  1. Mamba SSM block      — replaces self-attention (linear complexity, causal)
  2. Cross-attention      — standard attention to H-Net chunk embeddings
  3. FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.mamba = Mamba(
            d_model = d_model,
            d_state = 16,
            d_conv  = 4,
            expand  = 2,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        # 1. Mamba SSM — inherently causal
        tgt = tgt + self.mamba(self.norm1(tgt))

        # 2. Cross-attention to chunk embeddings
        tgt2, _ = self.cross_attn(
            query=self.norm2(tgt), key=memory, value=memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.drop2(tgt2)

        # 3. FFN
        tgt = tgt + self.drop3(self.ffn(self.norm3(tgt)))
        return tgt


class MambaByteDecoder(nn.Module):
    def __init__(
        self,
        d_model:     int   = 512,
        n_heads:     int   = 8,
        n_layers:    int   = 4,
        d_ff:        int   = None,
        dropout:     float = 0.1,
        max_tgt_len: int   = 512,
        enc_dim:     int   = 4096,
        vocab_size:  int   = 256,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.bos_id = vocab_size + 1   # 257
        self.eos_id = 1
        self.pad_id = 0

        self.tgt_embedding = nn.Embedding(vocab_size + 2, d_model, padding_idx=0)
        self.pos_encoding  = nn.Embedding(max_tgt_len, d_model)
        self.dropout       = nn.Dropout(dropout)
        self.encoder_proj  = nn.Linear(enc_dim, d_model, bias=False)

        self.layers = nn.ModuleList([
            MambaDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "mamba" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_ids, encoder_out, encoder_mask, tgt_key_padding_mask=None):
        B, T   = tgt_ids.shape
        device = tgt_ids.device

        pos    = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        tgt    = self.dropout(self.tgt_embedding(tgt_ids) + self.pos_encoding(pos))
        memory = self.encoder_proj(encoder_out)
        mem_pad_mask = ~encoder_mask.bool()

        for layer in self.layers:
            tgt = layer(tgt, memory, memory_key_padding_mask=mem_pad_mask)

        return self.output_proj(self.norm(tgt))

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_out,
        encoder_mask,
        max_len=300,
        device="cuda",
        # repetition_penalty=1.5,   # stronger than Transformer — Mamba loops more
        # no_repeat_ngram=3,        # block 3-gram repeats
    ):
        """
        Greedy decode with repetition penalty + n-gram blocking.
        Mamba needs stronger anti-repetition than Transformer decoder.
        """
        generated = [self.bos_id]

        for _ in range(max_len):
            tgt    = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self.forward(tgt, encoder_out, encoder_mask)
            scores = logits[0, -1].float()

            # # ── Repetition penalty ────────────────────────────────────────
            # seen = set(b for b in generated if 2 <= b <= 255)
            # for b in seen:
            #     if b < scores.shape[0]:
            #         scores[b] /= repetition_penalty

            # # ── No-repeat n-gram blocking ─────────────────────────────────
            # if len(generated) >= no_repeat_ngram:
            #     suffix = tuple(generated[-(no_repeat_ngram - 1):])
            #     for i in range(len(generated) - no_repeat_ngram + 1):
            #         if tuple(generated[i:i + no_repeat_ngram - 1]) == suffix:
            #             blocked = generated[i + no_repeat_ngram - 1]
            #             if 2 <= blocked <= 255:
            #                 scores[blocked] = float("-inf")

            nxt = scores.argmax(-1).item()

            if nxt in (self.eos_id, self.pad_id):
                break
            generated.append(nxt)

            # Hard loop guard — same byte > 10 times in last 20
            if len(generated) > 20:
                last20 = generated[-20:]
                if max(last20.count(b) for b in set(last20)) > 10:
                    break

        raw = [b for b in generated[1:] if 2 <= b <= 255]
        return bytes(raw).decode("utf-8", errors="ignore")