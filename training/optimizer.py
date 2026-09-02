# training/optimizer.py
"""
Optimizer & LR scheduler builder.

Four param groups with different LRs:
  1. LoRA params              → very small LR  (lr_backbone)
  2. Unfrozen backbone layers → very small LR  (lr_backbone * 0.5)
  3. H-Net params             → medium LR      (lr_hnet)
  4. Byte embedding + LM head → medium LR      (lr_lm)
"""

import math
import torch
from transformers import get_cosine_schedule_with_warmup


def build_optimizer_and_scheduler(
    *,
    encoder,
    lm_head,
    peft_model,
    num_samples:       int,
    batch_size:        int,
    grad_accum_steps:  int,
    num_epochs:        int,
    lr_backbone:       float = 5e-6,
    lr_hnet:           float = 5e-5,
    lr_lm:             float = 5e-5,
    weight_decay:      float = 0.01,
    warmup_ratio:      float = 0.05,
    extra_params:      list  = None,   # unfrozen backbone layer params
    extra_params_lr:   float = None,   # lr for extra params (default lr_backbone*0.5)
):
    # ── Parameter groups 

    # Group 1 — LoRA weights (all requires_grad params in peft_model)
    # lora_params = [p for p in peft_model.parameters() if p.requires_grad]
    
    extra_param_ids = {id(p) for p in (extra_params or [])}
    lora_params = [
        p for p in peft_model.parameters()
        if p.requires_grad and id(p) not in extra_param_ids
    ]

    # Group 2 — H-Net lightweight modules
    hnet_params = (
        list(encoder.routing.parameters()) +
        list(encoder.smoother.parameters()) +
        list(encoder.local_encoder.parameters()) +
        list(encoder.local_decoder.parameters())
    )

    # Group 3 — Byte embedding + LM head + skip proj
    lm_params = (
        list(encoder.byte_embedding.parameters()) +
        list(encoder.skip_proj.parameters()) +
        list(lm_head.parameters())
    )

    param_groups = []

    if lora_params:
        param_groups.append({
            "params":       lora_params,
            "lr":           lr_backbone,
            "weight_decay": weight_decay,
            "name":         "lora",
        })

    # Group 4 — Unfrozen backbone layers (first 3 + last 3)
    if extra_params:
        extra_lr = extra_params_lr or lr_backbone * 0.5
        param_groups.append({
            "params":       extra_params,
            "lr":           extra_lr,
            "weight_decay": weight_decay,
            "name":         "backbone_layers",
        })

    param_groups.append({
        "params":       hnet_params,
        "lr":           lr_hnet,
        "weight_decay": weight_decay,
        "name":         "hnet",
    })
    param_groups.append({
        "params":       lm_params,
        "lr":           lr_lm,
        "weight_decay": weight_decay,
        "name":         "lm",
    })

    optimizer = torch.optim.AdamW(param_groups)

    # Print param group summary
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  [{g.get('name','?'):15s}] lr={g['lr']:.2e}  params={n:,}")

    # ── Scheduler ─────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(num_samples / (batch_size * grad_accum_steps))
    total_steps     = steps_per_epoch * num_epochs
    warmup_steps    = max(1, int(warmup_ratio * total_steps))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    return optimizer, scheduler


def build_finetune_optimizer(
    *,
    task_head,
    encoder          = None,
    lr_head:  float  = 2e-4,
    lr_enc:   float  = 1e-5,
    weight_decay:    float = 0.01,
    num_steps:       int   = 1000,
    warmup_ratio:    float = 0.1,
):
    param_groups = [{"params": task_head.parameters(), "lr": lr_head}]

    if encoder is not None:
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        if enc_params:
            param_groups.append({"params": enc_params, "lr": lr_enc})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    warmup_steps = max(1, int(warmup_ratio * num_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = num_steps,
    )
    return optimizer, scheduler