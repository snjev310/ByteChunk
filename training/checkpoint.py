# training/checkpoint.py
"""
Fast async checkpoint saving.

Saves:
  - H-Net small modules (routing, smoother, embeddings, local enc/dec)
  - Unfrozen backbone layers (first 3 + last 3) — replaces LoRA in this experiment
  - LM head
  - Metadata
"""

import os
import time
import threading
import torch

_save_thread: threading.Thread = None


def save_checkpoint(
    path:        str,
    encoder,
    peft_model,            # backbone model (with or without LoRA)
    lm_head,
    optimizer,
    scheduler,
    global_step: int,
    best_loss:   float,
    config:      dict,
    task_head    = None,
    blocking:    bool = False,
):
    ckpt = _build_checkpoint(
        encoder, peft_model, lm_head,
        optimizer, scheduler,
        global_step, best_loss, config, task_head,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if blocking:
        _write_checkpoint(path, ckpt)
    else:
        _async_save(path, ckpt)


def _build_checkpoint(
    encoder, peft_model, lm_head,
    optimizer, scheduler,
    global_step, best_loss, config, task_head,
) -> dict:

    # ── LoRA weights (if LoRA is used) ────────────────────────────────────
    lora_state = {
        k: v.cpu() for k, v in peft_model.named_parameters()
        if "lora" in k and v.requires_grad
    }

    # ── Unfrozen backbone layer weights ───────────────────────────────────
    # Save any backbone layer params that require grad
    # This covers first 3 + last 3 layers in the no-LoRA experiment
    backbone_unfrozen = {}
    try:
        # Try to get llama layers — works for both peft_model and raw model
        if hasattr(peft_model, 'model') and hasattr(peft_model.model, 'model'):
            layers = peft_model.model.model.layers   # peft wrapped
        elif hasattr(peft_model, 'model') and hasattr(peft_model.model, 'layers'):
            layers = peft_model.model.layers          # raw AutoModelForCausalLM
        elif hasattr(peft_model, 'language_model') and hasattr(peft_model.language_model, 'layers'):
            layers = peft_model.language_model.layers  # Gemma3
        elif hasattr(peft_model, "model") and hasattr(peft_model.model, "layers"):
            layers = peft_model.model.layers  # Qwen3
        else:
            layers = None

        if layers is not None:
            for i, layer in enumerate(layers):
                layer_params = {
                    k: v.cpu() for k, v in layer.named_parameters()
                    if v.requires_grad
                }
                if layer_params:   # only save if this layer has trainable params
                    backbone_unfrozen[f"layer_{i}"] = layer_params
    except Exception as e:
        print(f"[checkpoint] Could not save backbone layers: {e}")

    ckpt = {
        # ── H-Net modules ─────────────────────────────────────────────────
        "byte_embedding": {k: v.cpu() for k, v in encoder.byte_embedding.state_dict().items()},
        "routing":        {k: v.cpu() for k, v in encoder.routing.state_dict().items()},
        "smoother":       {k: v.cpu() for k, v in encoder.smoother.state_dict().items()},
        "skip_proj":      {k: v.cpu() for k, v in encoder.skip_proj.state_dict().items()},
        "local_encoder":  {k: v.cpu() for k, v in encoder.local_encoder.state_dict().items()},
        "local_decoder":  {k: v.cpu() for k, v in encoder.local_decoder.state_dict().items()},

        # ── Backbone weights ───────────────────────────────────────────────
        "lora_state":           lora_state,            # empty dict if no LoRA
        "backbone_unfrozen":    backbone_unfrozen,      # unfrozen layer weights

        # ── LM head ───────────────────────────────────────────────────────
        "lm_head":        {k: v.cpu() for k, v in lm_head.state_dict().items()},

        # ── Metadata ──────────────────────────────────────────────────────
        "global_step":    global_step,
        "best_loss":      best_loss,
        "config":         config,
        "timestamp":      time.time(),
        "format_version": 4,   # bumped — now saves backbone_unfrozen
    }

    if task_head is not None:
        ckpt["task_head"] = {k: v.cpu() for k, v in task_head.state_dict().items()}

    n_backbone = sum(
        v.numel()
        for layer_dict in backbone_unfrozen.values()
        for v in layer_dict.values()
    )
    n_lora = sum(v.numel() for v in lora_state.values())
    print(f"  [ckpt] backbone_unfrozen={n_backbone/1e6:.1f}M  lora={n_lora/1e6:.1f}M")

    return ckpt


def _write_checkpoint(path: str, ckpt: dict):
    tmp = path + ".tmp"
    try:
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[checkpoint] Save failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)


def _async_save(path: str, ckpt: dict):
    global _save_thread
    if _save_thread is not None and _save_thread.is_alive():
        _save_thread.join()
    _save_thread = threading.Thread(
        target=_write_checkpoint,
        args=(path, ckpt),
        daemon=True,
    )
    _save_thread.start()


def wait_for_checkpoint():
    global _save_thread
    if _save_thread is not None and _save_thread.is_alive():
        print("[checkpoint] Waiting for background save to complete...")
        _save_thread.join()
        print("[checkpoint] Done.")


def load_checkpoint(path: str, device: str = "cuda") -> dict:
    return torch.load(path, map_location=device)