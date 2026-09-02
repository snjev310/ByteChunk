# training/trainer.py
"""
Pretraining step for H-Net.

Two step types:
  pretrain_step — standard AR + ratio + chunk alignment loss
  pos_step      — POS supervision loss on UD Hindi treebank batch

Interleaving: every POS_BATCH_FREQ AR steps, one POS step is run.
This gives the encoder direct linguistic supervision without
dominating the AR pretraining signal.
"""

import torch
import math
from models.hnet_loss import chunk_alignment_loss


def pretrain_step(
    batch:                dict,
    encoder,
    lm_head,
    hnet_loss_fn,
    optimizer,
    scheduler,
    global_step:          int,
    grad_accum_steps:     int,
    max_grad_norm:        float,
    device:               torch.device,
    lambda_align:         float = 0.0,
    byte_align_targets:   torch.Tensor = None,
) -> dict:

    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # ── Forward ───────────────────────────────────────────────────────────
    h_hat, p, aux  = encoder(input_ids, attention_mask)
    logits         = lm_head(h_hat)

    # ── AR + ratio loss ───────────────────────────────────────────────────
    total_loss, loss_ar, loss_ratio, actual_ratio = hnet_loss_fn(
        logits, input_ids, attention_mask, p
    )
    bpb = loss_ar / math.log(2)

    # ── Early bad batch detection ─────────────────────────────────────────
    if loss_ar.item() > 8.3 or torch.isnan(loss_ar) or torch.isinf(loss_ar):
        optimizer.zero_grad()
        return _bad_return(actual_ratio)

    # ── Chunk alignment loss ──────────────────────────────────────────────
    loss_align = torch.tensor(0.0, device=device)
    if lambda_align > 0.0 and byte_align_targets is not None:
        chunk_out        = aux["chunk_out"]
        chunk_embs_batch = chunk_out["chunk_embs"]
        chunk_ids_batch  = chunk_out["chunk_ids"]
        smoothed         = chunk_out.get("chunk_embs_smooth", chunk_embs_batch)
        loss_align = chunk_alignment_loss(
            chunk_embs_batch   = smoothed,
            chunk_ids_batch    = chunk_ids_batch,
            input_ids          = input_ids,
            attention_mask     = attention_mask,
            byte_align_targets = byte_align_targets,
        )
        if torch.isnan(loss_align) or torch.isinf(loss_align):
            optimizer.zero_grad()
            return _bad_return(actual_ratio)
        total_loss = total_loss + lambda_align * loss_align

    # ── Final NaN check ───────────────────────────────────────────────────
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        optimizer.zero_grad()
        return _bad_return(actual_ratio)

    # ── Backward ──────────────────────────────────────────────────────────
    (total_loss / grad_accum_steps).backward()

    did_step = False
    if (global_step + 1) % grad_accum_steps == 0:
        all_params = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        did_step = True

    # ── Diagnostics ───────────────────────────────────────────────────────
    b_hard = aux["b_hard"]
    with torch.no_grad():
        valid   = attention_mask.bool()
        p_valid = p[valid]
        p_stats = {
            "mean": p_valid.mean().item(),
            "std":  p_valid.std().item(),
            "min":  p_valid.min().item(),
            "max":  p_valid.max().item(),
        }
        num_chunks_list = aux["num_chunks"]
        valid_tokens    = attention_mask.float().sum(dim=1)
        nc_t            = torch.tensor(num_chunks_list, dtype=torch.float, device=device)
        avg_len         = (valid_tokens / nc_t.clamp(min=1)).mean().item()
        chunk_stats     = {
            "avg_num_chunks":   nc_t.mean().item(),
            "avg_chunk_length": avg_len,
        }

    if global_step % 500 == 0:
        ratio_p = (p * attention_mask).sum(1) / attention_mask.float().sum(1).clamp(1)
        ratio_b = (b_hard * attention_mask).sum(1) / attention_mask.float().sum(1).clamp(1)
        print(
            f"[step {global_step}] "
            f"p_ratio={ratio_p.mean():.3f}  b_ratio={ratio_b.mean():.3f}  "
            f"sim_mean={aux['router_out'].sim.mean():.3f}  "
            f"align={loss_align.item():.4f}"
        )

    return {
        "total_loss":         total_loss.detach(),
        "loss_ar":            loss_ar.detach(),
        "loss_ratio":         loss_ratio.detach(),
        "loss_align":         loss_align.detach(),
        "bpb":                bpb.detach(),
        "avg_boundary_ratio": actual_ratio.mean().detach(),
        "p_stats":            p_stats,
        "chunk_stats":        chunk_stats,
        "did_step":           did_step,
    }


def pos_step(
    pos_batch:       dict,
    encoder,
    pos_head,
    optimizer,
    scheduler,
    global_step:     int,
    grad_accum_steps: int,
    max_grad_norm:   float,
    device:          torch.device,
    lambda_pos:      float = 0.3,
) -> dict:
    """
    One POS supervision step.

    Runs encoder on a POS-labeled Hindi sentence batch,
    computes cross-entropy loss on word-level tags,
    and updates encoder + pos_head weights.

    This directly forces the encoder to produce
    linguistically discriminative representations.
    """
    input_ids      = pos_batch["input_ids"].to(device)
    attention_mask = pos_batch["attention_mask"].to(device)
    labels         = pos_batch["labels"].to(device)

    # Encoder forward — with gradients (not frozen during pretraining)
    h_hat, p, aux = encoder(input_ids, attention_mask)

    # POS head forward + loss
    logits   = pos_head(h_hat)                            # [B, T, n_tags]
    loss_pos = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index = -100,
    )

    if torch.isnan(loss_pos) or torch.isinf(loss_pos):
        optimizer.zero_grad()
        return {"loss_pos": torch.tensor(0.0), "pos_acc": 0.0, "did_step": False}

    total_loss = lambda_pos * loss_pos

    # Compute accuracy for logging
    with torch.no_grad():
        preds   = logits.argmax(-1).reshape(-1)
        targets = labels.reshape(-1)
        mask    = targets != -100
        pos_acc = (preds[mask] == targets[mask]).float().mean().item() if mask.sum() > 0 else 0.0

    # Backward
    (total_loss / grad_accum_steps).backward()

    did_step = False
    if (global_step + 1) % grad_accum_steps == 0:
        all_params = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        did_step = True

    return {
        "loss_pos": loss_pos.detach(),
        "pos_acc":  pos_acc,
        "did_step": did_step,
    }


def _bad_return(actual_ratio):
    return {
        "total_loss":         torch.tensor(0.0),
        "loss_ar":            torch.tensor(0.0),
        "loss_ratio":         torch.tensor(0.0),
        "loss_align":         torch.tensor(0.0),
        "bpb":                torch.tensor(0.0),
        "avg_boundary_ratio": actual_ratio.mean().detach(),
        "p_stats":            {"mean":0,"std":0,"min":0,"max":0},
        "chunk_stats":        {"avg_num_chunks":0,"avg_chunk_length":0},
        "did_step":           False,
    }