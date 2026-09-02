# models/hnet_loss.py
"""
H-Net pretraining loss (paper §2.3) + chunk-level alignment loss.

Three components:
  1. Autoregressive byte-level LM loss  (cross-entropy, shift by 1)
  2. Boundary ratio regularization      ((actual_ratio - target_ratio)^2)
  3. Chunk-level alignment loss         (cosine distance: chunk_emb ↔ subword_emb)

The chunk-level alignment loss is the key fix:
  - Operates AFTER chunking (not on raw bytes)
  - Each chunk ≈ one Hindi word
  - Target = average of byte_alignment_targets for bytes in that chunk
  - Forces chunk embeddings into subword embedding space
  - Much more meaningful than byte-level alignment

Total: L = L_ar + λ_ratio * L_ratio + λ_align * L_align
"""

import torch
import torch.nn.functional as F


def hnet_loss(
    logits:         torch.Tensor,   # [B, T, 256]
    input_ids:      torch.Tensor,   # [B, T]
    attention_mask: torch.Tensor,   # [B, T]
    p:              torch.Tensor,   # [B, T]
    target_ratio:   float = 0.125,
    lambda_ratio:   float = 0.5,
) -> tuple:
    # ── 1. AR loss ────────────────────────────────────────────────────────
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask   = attention_mask[:, 1:]
    shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
    loss_ar = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )

    # ── 2. Boundary ratio loss ────────────────────────────────────────────
    valid_p      = p * attention_mask.float()
    seq_lens     = attention_mask.float().sum(dim=1).clamp(min=1)
    actual_ratio = valid_p.sum(dim=1) / seq_lens
    loss_ratio   = ((actual_ratio - target_ratio) ** 2).mean()

    total = loss_ar + lambda_ratio * loss_ratio
    return total, loss_ar, loss_ratio, actual_ratio.detach()


def chunk_alignment_loss(
    chunk_embs_batch:    list,          # List[Tensor[C_i, D]] — smoothed chunk embs
    chunk_ids_batch:     list,          # List[Tensor[T_i]]    — which chunk each byte belongs to
    input_ids:           torch.Tensor,  # [B, T]               — byte ids
    attention_mask:      torch.Tensor,  # [B, T]
    byte_align_targets:  torch.Tensor,  # [256, D]             — precomputed byte→subword targets
) -> torch.Tensor:
    """
    Chunk-level alignment loss.

    For each chunk (≈ one Hindi word):
      1. Find which bytes belong to this chunk
      2. Look up their byte_align_targets (precomputed subword-space vectors)
      3. Average to get the chunk's target in subword space
      4. Minimise cosine distance: chunk_emb ↔ target

    This operates AFTER chunking so targets are word-level, not byte-level.
    Much more meaningful than byte-level alignment because:
      - Chunks correspond to actual words
      - Target = average subword embedding of that word's bytes
      - Directly bridges chunk space to subword space at word granularity
    """
    B = len(chunk_embs_batch)
    device = chunk_embs_batch[0].device
    dtype  = chunk_embs_batch[0].dtype

    total_loss  = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_count = 0

    byte_targets = byte_align_targets.to(device=device, dtype=torch.float32)

    for i in range(B):
        chunk_embs = chunk_embs_batch[i]          # [C_i, D]
        chunk_ids  = chunk_ids_batch[i]            # [T_i] — chunk id per byte
        T_i        = int(attention_mask[i].sum().item())
        byte_ids_i = input_ids[i, :T_i]            # [T_i] byte values

        C_i = chunk_embs.shape[0]

        # For each chunk, collect the bytes belonging to it
        # and average their subword targets
        chunk_targets = torch.zeros(C_i, byte_targets.shape[1],
                                    device=device, dtype=torch.float32)
        chunk_counts  = torch.zeros(C_i, device=device, dtype=torch.float32)

        cids_clipped = chunk_ids[:T_i].clamp(0, C_i - 1)  # [T_i]
        
        # AFTER — uses safe length
        T_i_safe     = min(T_i, len(chunk_ids))
        cids_clipped = chunk_ids[:T_i_safe].clamp(0, C_i - 1)
        # for t in range(T_i_safe):

        for t in range(T_i_safe):
            c_idx    = cids_clipped[t].item()
            byte_val = byte_ids_i[t].item()
            if 0 <= byte_val < 256:
                chunk_targets[c_idx] += byte_targets[byte_val]
                chunk_counts[c_idx]  += 1

        # Average — chunks with no bytes get zero target (skip)
        valid_chunks = chunk_counts > 0
        if valid_chunks.sum() == 0:
            continue

        chunk_counts_safe = chunk_counts.clamp(min=1).unsqueeze(1)
        chunk_targets     = chunk_targets / chunk_counts_safe   # [C_i, D]

        # Normalise both for cosine similarity
        embs_norm    = F.normalize(chunk_embs.float(), dim=-1)   # [C_i, D]
        targets_norm = F.normalize(chunk_targets, dim=-1)         # [C_i, D]

        # Cosine distance = 1 - cosine_similarity
        cos_sim = (embs_norm * targets_norm).sum(-1)              # [C_i]
        loss_i  = (1.0 - cos_sim)[valid_chunks]                   # only valid chunks

        total_loss  = total_loss + loss_i.sum()
        total_count += valid_chunks.sum().item()

    if total_count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / total_count


def alignment_loss(
    byte_embeddings:    torch.Tensor,
    llama_embed_table:  torch.Tensor,
    attention_mask:     torch.Tensor,
) -> torch.Tensor:
    """Original byte-level alignment loss — kept for reference."""
    B, T, D   = byte_embeddings.shape
    byte_norm  = F.normalize(byte_embeddings.float(), dim=-1)
    llama_norm = F.normalize(llama_embed_table.float(), dim=-1)

    with torch.no_grad():
        chunk_size = 4096
        V          = llama_norm.shape[0]
        best_sim   = torch.full((B, T), -1.0, device=byte_embeddings.device)
        best_idx   = torch.zeros((B, T), dtype=torch.long, device=byte_embeddings.device)

        for start in range(0, V, chunk_size):
            end  = min(start + chunk_size, V)
            sim  = torch.matmul(byte_norm, llama_norm[start:end].T)
            chunk_best_sim, chunk_best_idx = sim.max(-1)
            chunk_best_idx = chunk_best_idx + start
            update_mask    = chunk_best_sim > best_sim
            best_sim       = torch.where(update_mask, chunk_best_sim, best_sim)
            best_idx       = torch.where(update_mask, chunk_best_idx, best_idx)

        targets = llama_norm[best_idx]

    cos_sim = (byte_norm * targets).sum(-1)
    loss    = 1.0 - cos_sim
    mask    = attention_mask.float()
    return (loss * mask).sum() / mask.sum().clamp(min=1)