# tasks/task_heads.py
"""
Task-specific heads that sit on top of HNetEncoder.

All heads receive h_hat [B, T, D] — byte-resolution context vectors
from the encoder — and produce task-specific outputs.

Three tasks:
  1. MT / Language Modeling  → ByteLMHead    [B, T, 256]
  2. POS Tagging             → POSHead       [B, T, n_tags]
  3. Sentiment Analysis      → SentimentHead [B, n_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. MT / LM head ───────────────────────────────────────────────────────────

class ByteLMHead(nn.Module):
    """Autoregressive byte prediction head. Used for pretraining and MT."""

    def __init__(self, d_model: int, vocab_size: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, D]  →  logits: [B, T, 256]"""
        return self.proj(h)

    def compute_loss(self, h, labels, attention_mask) -> torch.Tensor:
        logits       = self.forward(h)
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        pad_mask     = attention_mask[:, 1:] == 0
        shift_labels = shift_labels.masked_fill(pad_mask, -100)
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )


# ── 2. POS tagging head ───────────────────────────────────────────────────────

class POSHead(nn.Module):
    """
    Byte-level sequence labeling head.
    Predicts a POS tag for each byte position.

    Uses a bottleneck projection (d_model → 256 → n_tags) instead of
    direct d_model → n_tags. This is critical for d_model=4096 where a
    direct linear head has too few parameters to learn useful features.
    """

    def __init__(self, d_model: int, n_tags: int, dropout: float = 0.1):
        super().__init__()
        hidden          = min(256, d_model // 4)   # 256 for d_model=4096
        self.dropout    = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(d_model, hidden, bias=True)
        self.act        = nn.GELU()
        self.norm       = nn.LayerNorm(hidden)
        self.proj       = nn.Linear(hidden, n_tags, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, D]  →  logits: [B, T, n_tags]"""
        h = self.dropout(h)
        h = self.act(self.bottleneck(h))   # [B, T, 256]
        h = self.norm(h)
        return self.proj(h)                # [B, T, n_tags]

    def compute_loss(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.forward(h)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def predicts(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h).argmax(dim=-1)

    @torch.no_grad()
    def word_accuracy(self, h: torch.Tensor, labels: torch.Tensor) -> float:
        preds   = self.predicts(h).reshape(-1)
        targets = labels.reshape(-1)
        mask    = targets != -100
        if mask.sum() == 0:
            return 0.0
        correct = (preds[mask] == targets[mask]).sum().item()
        total   = mask.sum().item()
        return correct / total


# ── 3. Sentiment head ─────────────────────────────────────────────────────────

class SentimentHead(nn.Module):
    """
    Sequence-level classification over byte representations.
    Supports three pooling strategies:
      "mean" — average over non-padding positions
      "last" — take the last non-padding position
      "cls"  — take position 0 (like BERT [CLS])
    """

    def __init__(self, d_model: int, n_classes: int,
                 pool: str = "mean", dropout: float = 0.1):
        super().__init__()
        assert pool in ("mean", "last", "cls"), f"Invalid pool: {pool}"
        self.pool    = pool
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)
        self.proj    = nn.Linear(d_model, n_classes)

    def _pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            mask_f = attention_mask.float().unsqueeze(-1)
            return (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        elif self.pool == "last":
            lengths = (attention_mask.sum(dim=1) - 1).clamp(min=0)
            idx     = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
            return h.gather(1, idx).squeeze(1)
        else:   # cls
            return h[:, 0, :]

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(h, attention_mask)
        pooled = self.dropout(self.norm(pooled))
        return self.proj(pooled)

    def compute_loss(self, h, attention_mask, labels) -> torch.Tensor:
        return F.cross_entropy(self.forward(h, attention_mask), labels)

    @torch.no_grad()
    def predicts(self, h, attention_mask) -> torch.Tensor:
        return self.forward(h, attention_mask).argmax(dim=-1)

    @torch.no_grad()
    def accuracy(self, h, attention_mask, labels) -> float:
        preds   = self.predicts(h, attention_mask)
        correct = (preds == labels).sum().item()
        return correct / labels.size(0) if labels.size(0) > 0 else 0.0