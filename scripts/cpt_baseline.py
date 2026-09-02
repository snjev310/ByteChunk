# scripts/cpt_baseline.py
"""
Continued Pretraining (CPT) baseline for Qwen3-4B on Hindi data.
Same setup as H-Net pretraining:
  - Same 500K Hindi sentences from FineWeb-2
  - Same unfrozen layers (first + last transformer layers)
  - Same optimizer, LR, batch size
  - Standard subword AR loss only

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.cpt_baseline
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

from configs.default import (
    MODEL_ID, HIDDEN_SIZE, DEVICE, DTYPE,
    BATCH_SIZE, GRAD_ACCUM_STEPS, LR, WEIGHT_DECAY,
    MAX_GRAD_NORM, WARMUP_RATIO, SEED,
)

# ── CPT-specific config ───────────────────────────────────────────────────────
CPT_SAVE_DIR    = f"runs_cpt/{MODEL_ID.replace('/', '_')}"
CPT_MAX_LENGTH  = 512   # subword tokens (not bytes)
CPT_NUM_STEPS   = 62500 # same as H-Net
CPT_SAVE_EVERY  = 500
CPT_LOG_EVERY   = 100
N_UNFREEZE      = 1     # unfreeze first and last N layers

os.makedirs(CPT_SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)

# ── Load model and tokenizer ──────────────────────────────────────────────────
print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"  Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ── Freeze all layers except first and last N ─────────────────────────────────
for param in model.parameters():
    param.requires_grad = False

# Get transformer layers
llama = getattr(model, 'language_model', None) or getattr(model, 'model', model)
layers = llama.layers
n_layers = len(layers)
print(f"  Total layers: {n_layers}")

# Unfreeze first N and last N layers
unfreeze_indices = list(range(N_UNFREEZE)) + list(range(n_layers - N_UNFREEZE, n_layers))
for idx in unfreeze_indices:
    for param in layers[idx].parameters():
        param.requires_grad = True

# Also unfreeze LM head
for param in model.lm_head.parameters():
    param.requires_grad = True

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {n_trainable/1e6:.1f}M")
print(f"  Unfrozen layers: {unfreeze_indices}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class HindiSubwordDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.texts      = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc  = self.tokenizer(
            text,
            max_length      = self.max_length,
            truncation      = True,
            padding         = "max_length",
            return_tensors  = "pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels: shift right, mask padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

# ── Stream Hindi data ─────────────────────────────────────────────────────────
print("Streaming Hindi data from FineWeb-2...")
ds = load_dataset(
    "HuggingFaceFW/fineweb-2",
    "hin_Deva",
    split     = "train",
    streaming = True,
).shuffle(seed=SEED)

texts = []
for item in tqdm(ds, desc="Loading 500K sentences", total=500_000):
    texts.append(item["text"])
    if len(texts) >= 500_000:
        break

print(f"  Loaded {len(texts):,} sentences")

dataset    = HindiSubwordDataset(texts, tokenizer, CPT_MAX_LENGTH)
dataloader = DataLoader(
    dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 0,
    pin_memory  = True,
)

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr           = LR,
    weight_decay = WEIGHT_DECAY,
)

total_steps   = CPT_NUM_STEPS
warmup_steps  = int(total_steps * WARMUP_RATIO)
scheduler     = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = warmup_steps,
    num_training_steps = total_steps,
)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting CPT for {total_steps} steps...")
print(f"  Batch size:      {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Max length:      {CPT_MAX_LENGTH} subword tokens")
print(f"  Save dir:        {CPT_SAVE_DIR}")

model.train()
global_step  = 0
best_loss    = float("inf")
accum_loss   = 0.0
accum_steps  = 0
data_iter    = iter(dataloader)

pbar = tqdm(total=total_steps, desc="CPT")

optimizer.zero_grad()

while global_step < total_steps:
    # Get next batch (cycle through dataset)
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch     = next(data_iter)

    input_ids      = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels         = batch["labels"].to(DEVICE)

    # Forward pass
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )
        loss = outputs.loss / GRAD_ACCUM_STEPS

    loss.backward()
    accum_loss  += loss.item()
    accum_steps += 1

    if accum_steps == GRAD_ACCUM_STEPS:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            MAX_GRAD_NORM,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        step_loss    = accum_loss
        step_bpb     = step_loss / math.log(2)
        accum_loss   = 0.0
        accum_steps  = 0

        pbar.update(1)
        pbar.set_postfix(loss=f"{step_loss:.4f}", bpb=f"{step_bpb:.4f}")

        # Log
        if global_step % CPT_LOG_EVERY == 0:
            print(f"\n[step {global_step:6d}] loss={step_loss:.4f}  bpb={step_bpb:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if step_loss < best_loss:
            best_loss = step_loss
            ckpt = {
                "model_state": {
                    k: v for k, v in model.state_dict().items()
                    if any(f"layers.{i}." in k for i in unfreeze_indices)
                    or "lm_head" in k
                },
                "global_step": global_step,
                "best_loss":   best_loss,
                "config": {
                    "model_id":        MODEL_ID,
                    "hidden_size":     HIDDEN_SIZE,
                    "max_length":      CPT_MAX_LENGTH,
                    "n_unfreeze":      N_UNFREEZE,
                    "unfreeze_layers": unfreeze_indices,
                },
            }
            torch.save(ckpt, os.path.join(CPT_SAVE_DIR, "best.pt"))

        # Save periodic checkpoint
        if global_step % CPT_SAVE_EVERY == 0:
            ckpt = {
                "model_state": {
                    k: v for k, v in model.state_dict().items()
                    if any(f"layers.{i}." in k for i in unfreeze_indices)
                    or "lm_head" in k
                },
                "global_step": global_step,
                "best_loss":   best_loss,
                "config": {
                    "model_id":        MODEL_ID,
                    "hidden_size":     HIDDEN_SIZE,
                    "max_length":      CPT_MAX_LENGTH,
                    "n_unfreeze":      N_UNFREEZE,
                    "unfreeze_layers": unfreeze_indices,
                },
            }
            torch.save(ckpt, os.path.join(CPT_SAVE_DIR, f"step_{global_step}.pt"))
            print(f"  [ckpt] Saved step_{global_step}.pt  best_loss={best_loss:.4f}")

pbar.close()
print(f"\nCPT complete.")
print(f"  Best loss: {best_loss:.4f}  BPB: {best_loss/math.log(2):.4f}")
print(f"  Saved to:  {CPT_SAVE_DIR}/")