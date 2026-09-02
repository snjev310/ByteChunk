# models/load_hnet.py
"""
Load a pretrained H-Net checkpoint into a fully constructed HNetEncoder.

Supports two checkpoint formats:
  format_version=3: LoRA weights in "lora_state"
  format_version=4: Unfrozen backbone layers in "backbone_unfrozen" (no LoRA)
"""

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

from models.routing      import RoutingModule
from models.chunking     import DynamicChunking
from models.smoothing    import SmoothingModule
from models.upsample     import EMADeChunk
from models.hnet_encoder import HNetEncoder
from backbone.lm_head    import ByteLMHead


def load_hnet_encoder(
    checkpoint_path: str,
    model_id:        str         = "meta-llama/Llama-3.1-8B",
    device:          str         = "cuda",
    dtype:           torch.dtype = torch.bfloat16,
    frozen:          bool        = True,
    load_lm_head:    bool        = True,
    n_local_layers:  int         = 3,
) -> tuple:
    """
    Returns:
        encoder  : HNetEncoder (weights loaded, optionally frozen)
        lm_head  : ByteLMHead  (None if load_lm_head=False)
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Use model_id from checkpoint config if available
    ckpt_model_id = ckpt.get("config", {}).get("model_id", None)
    if ckpt_model_id and ckpt_model_id != model_id:
        print(f"  Overriding model_id: {model_id} → {ckpt_model_id}")
        model_id = ckpt_model_id

    hidden_size    = ckpt["byte_embedding"]["weight"].shape[1]
    fmt_version    = ckpt.get("format_version", 3)
    has_lora       = bool(ckpt.get("lora_state", {}))
    has_unfrozen   = bool(ckpt.get("backbone_unfrozen", {}))

    print(f"  hidden_size     = {hidden_size}")
    print(f"  step            = {ckpt.get('global_step', '?')}")
    print(f"  best_loss       = {ckpt.get('best_loss', '?')}")
    print(f"  format_version  = {fmt_version}")
    print(f"  has_lora        = {has_lora}")
    print(f"  has_unfrozen    = {has_unfrozen}")

    # ── H-Net small modules ───────────────────────────────────────────────
    byte_embedding = nn.Embedding(256, hidden_size, padding_idx=0)
    byte_embedding.load_state_dict(ckpt["byte_embedding"])
    byte_embedding.to(device=device, dtype=dtype)

    routing = RoutingModule(hidden_size)
    routing.load_state_dict(ckpt["routing"])
    routing.to(device=device, dtype=dtype)

    smoother = SmoothingModule(hidden_size)
    smoother.load_state_dict(ckpt["smoother"])
    smoother.to(device=device, dtype=dtype)

    chunker     = DynamicChunking()
    ema_dechunk = EMADeChunk()

    # ── Backbone loading — auto-detect format ─────────────────────────────
    print("  Loading backbone...")
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype      = dtype,
        device_map = "auto",
    )

    if has_unfrozen:
        # ── Format v4: unfrozen backbone layers (no LoRA) ─────────────────
        backbone_unfrozen = ckpt["backbone_unfrozen"]
        n_loaded = 0
        for layer_key, layer_state in backbone_unfrozen.items():
            # layer_key = "layer_0", "layer_1", etc.
            layer_idx = int(layer_key.split("_")[1])
            layer     = getattr(base, "language_model", base.model).layers[layer_idx]
            # Load each param by name
            for param_name, param_val in layer_state.items():
                parts  = param_name.split(".")
                module = layer
                for part in parts[:-1]:
                    module = getattr(module, part)
                getattr(module, parts[-1]).data.copy_(
                    param_val.to(device=device, dtype=dtype)
                )
                n_loaded += 1
        print(f"  Backbone unfrozen layers loaded: {list(backbone_unfrozen.keys())}  ({n_loaded} tensors)")
        llama = getattr(base, "language_model", None) or getattr(base, "model", base)  # inner model

    else:
        # ── Format v3: LoRA weights ───────────────────────────────────────
        lora_config = LoraConfig(
            r              = 64,
            lora_alpha     = 128,
            target_modules = ["q_proj", "k_proj", "v_proj"],
            lora_dropout   = 0.05,
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base, lora_config)

        lora_state = ckpt.get("lora_state", {})
        if lora_state:
            model_keys   = set(peft_model.state_dict().keys())
            missing, unexpected = peft_model.load_state_dict(lora_state, strict=False)

            if len(missing) > 10:
                fixed = {}
                for k, v in lora_state.items():
                    if k in model_keys:
                        fixed[k] = v
                        continue
                    for prefix in ["base_model.model.", "base_model.", "model."]:
                        if prefix + k in model_keys:
                            fixed[prefix + k] = v
                            break
                    else:
                        for prefix in ["base_model.model.", "base_model.", "model."]:
                            if k.startswith(prefix):
                                stripped = k[len(prefix):]
                                if stripped in model_keys:
                                    fixed[stripped] = v
                                    break
                if fixed:
                    missing, unexpected = peft_model.load_state_dict(fixed, strict=False)
                    print(f"  LoRA loaded after prefix fix (missing={len(missing)})")
                else:
                    print(f"  Warning: LoRA prefix fix failed")
            else:
                print(f"  LoRA loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("  Warning: no LoRA weights found — using base backbone")

        llama = peft_model.model.model   # inner LlamaModel

    # ── Assemble HNetEncoder ──────────────────────────────────────────────
    encoder = HNetEncoder(
        byte_embedding  = byte_embedding,
        routing         = routing,
        chunker         = chunker,
        smoother        = smoother,
        ema_dechunk     = ema_dechunk,
        llama           = llama,
        d_model         = hidden_size,
        n_local_layers  = n_local_layers,
    )

    if "skip_proj" in ckpt:
        encoder.skip_proj.load_state_dict(ckpt["skip_proj"])

    if "local_encoder" in ckpt:
        encoder.local_encoder.load_state_dict(ckpt["local_encoder"])

    if "local_decoder" in ckpt:
        encoder.local_decoder.load_state_dict(ckpt["local_decoder"])

    # Move only non-backbone parts — backbone already placed by device_map=auto
    encoder.byte_embedding.to(device=device, dtype=dtype)
    encoder.routing.to(device=device, dtype=dtype)
    encoder.smoother.to(device=device, dtype=dtype)
    encoder.ema_dechunk.to(device=device, dtype=dtype)
    encoder.skip_proj.to(device=device, dtype=dtype)
    if hasattr(encoder, "local_encoder"):
        encoder.local_encoder.to(device=device, dtype=dtype)
    if hasattr(encoder, "local_decoder"):
        encoder.local_decoder.to(device=device, dtype=dtype)

    if frozen:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        print("  Encoder frozen for fine-tuning")

    # ── LM head ───────────────────────────────────────────────────────────
    lm_head = None
    if load_lm_head and "lm_head" in ckpt:
        lm_head = ByteLMHead(hidden_dim=hidden_size, vocab_size=256)
        lm_head.load_state_dict(ckpt["lm_head"])
        lm_head.to(device=device, dtype=dtype)
        if frozen:
            for param in lm_head.parameters():
                param.requires_grad = False
        print("  LM head loaded")

    return encoder, lm_head