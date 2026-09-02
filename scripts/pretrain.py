# scripts/pretrain.py
"""
H-Net pretraining with task-guided POS supervision.

Every POS_BATCH_FREQ AR steps, one POS supervision step is run.
This interleaving forces the encoder to learn linguistically
structured representations alongside byte-level prediction.
"""
import os
import torch
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from configs.default import *
from data.pretrain_dataset import ByteDataset, ByteTokenizer, load_monolingual_data
from data.pos_dataset import get_pos_dataloaders
from models.routing import RoutingModule
from models.chunking import DynamicChunking
from models.smoothing import SmoothingModule
from models.upsample import EMADeChunk
from models.hnet_encoder import HNetEncoder
from models.hnet_loss import hnet_loss
from tasks.task_heads import POSHead
from backbone.lm_head import ByteLMHead
from training.trainer import pretrain_step, pos_step
from training.optimizer import build_optimizer_and_scheduler
from training.checkpoint import save_checkpoint, wait_for_checkpoint
from utils.logging import init_loss_history, update_loss_history, save_loss_history

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(SEED)
os.makedirs(PRETRAIN_SAVE_DIR, exist_ok=True)

# ── AR pretraining data ───────────────────────────────────────────────────────
print("Loading monolingual data...")
texts     = load_monolingual_data(split="train", language="hi", n=500000)
tokenizer = ByteTokenizer(max_length=MAX_LENGTH)
dataset   = ByteDataset(texts, tokenizer)
# loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"  AR dataset size: {len(dataset)}")

# ── POS supervision data ──────────────────────────────────────────────────────
print("Loading POS supervision data...")
# pos_train_loader, _, pos_tag2id, pos_n_tags = get_pos_dataloaders(
#     conll_path = POS_CONLL_PATH,
#     max_len    = 512,
#     batch_size = BATCH_SIZE,
#     num_workers= 2,
# )

pos_train_loader, _, pos_tag2id, pos_n_tags = get_pos_dataloaders(
    conll_path  = POS_CONLL_PATH,
    max_len     = 512,
    batch_size  = BATCH_SIZE,
    num_workers = 0,    # was 2
)

pos_iter = itertools.cycle(pos_train_loader)  # cycle so it never runs out
print(f"  POS tags: {pos_n_tags}  |  POS batches/epoch: {len(pos_train_loader)}")

# ── Byte alignment targets ────────────────────────────────────────────────────
byte_align_targets = None
target_files = {
    "Qwen/Qwen2.5-1.5B":    "byte_alignment_targets.pt",
    "Qwen/Qwen3-4B":         "byte_alignment_targets_qwen3_4b.pt",
    "google/gemma-3-4b-pt":  "byte_alignment_targets_gemma.pt",
    "google/gemma-3-12b-pt": "byte_alignment_targets_gemma12b.pt",
}
target_file = target_files.get(MODEL_ID, None)
if target_file and os.path.exists(target_file):
    byte_align_targets = torch.load(target_file).to(DEVICE)
    print(f"  Loaded {target_file}")
else:
    print(f"  WARNING: No byte alignment targets for {MODEL_ID}")

# ── Build H-Net modules ───────────────────────────────────────────────────────
byte_embedding = torch.nn.Embedding(
    BYTE_VOCAB_SIZE, HIDDEN_SIZE, padding_idx=PAD_ID
).to(DEVICE, dtype=DTYPE)

if byte_align_targets is not None:
    with torch.no_grad():
        byte_embedding.weight.data.copy_(byte_align_targets[:BYTE_VOCAB_SIZE].to(DTYPE))
    print(f"  Byte embeddings initialised from subword targets")

routing     = RoutingModule(HIDDEN_SIZE).to(DEVICE, dtype=DTYPE)
chunker     = DynamicChunking().to(DEVICE)
smoother    = SmoothingModule(HIDDEN_SIZE).to(DEVICE, dtype=DTYPE)
ema_dechunk = EMADeChunk().to(DEVICE)

# ── Backbone — freeze all, unfreeze first+last N layers ───────────────────────
print(f"Loading backbone: {MODEL_ID}")
base  = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
# llama = base.model

llama = getattr(base, 'language_model', None) or getattr(base, 'model', base)

for param in llama.parameters():
    param.requires_grad = False

n_layers      = len(llama.layers)
unfreeze_idxs = list(range(N_UNFREEZE_LAYERS)) + \
                list(range(n_layers - N_UNFREEZE_LAYERS, n_layers))

for i in unfreeze_idxs:
    for param in llama.layers[i].parameters():
        param.requires_grad = True

unfrozen_backbone_params = [
    p for i, layer in enumerate(llama.layers)
    if i in unfreeze_idxs
    for p in layer.parameters()
    if p.requires_grad
]
print(f"  Unfrozen layers: {unfreeze_idxs}")
print(f"  Unfrozen params: {sum(p.numel() for p in unfrozen_backbone_params)/1e6:.1f}M")

# ── HNetEncoder ───────────────────────────────────────────────────────────────
encoder = HNetEncoder(
    byte_embedding  = byte_embedding,
    routing         = routing,
    chunker         = chunker,
    smoother        = smoother,
    ema_dechunk     = ema_dechunk,
    llama           = llama,
    d_model         = HIDDEN_SIZE,
    boundary_thresh = BOUNDARY_THRESH,
    n_local_layers  = N_LOCAL_LAYERS,
).to(DEVICE, dtype=DTYPE)

lm_head = ByteLMHead(HIDDEN_SIZE, BYTE_VOCAB_SIZE).to(DEVICE, dtype=DTYPE)

# ── POS head (trained jointly during pretraining) ─────────────────────────────
pos_head = POSHead(d_model=HIDDEN_SIZE, n_tags=pos_n_tags, dropout=0.1)
pos_head = pos_head.to(DEVICE, dtype=DTYPE)
torch.save(pos_tag2id, os.path.join(PRETRAIN_SAVE_DIR, "pos_tag2id.pt"))
print(f"  POS head: {HIDDEN_SIZE} → {pos_n_tags} tags")

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer, scheduler = build_optimizer_and_scheduler(
    encoder               = encoder,
    lm_head               = lm_head,
    peft_model            = base,
    num_samples           = len(dataset),
    batch_size            = BATCH_SIZE,
    grad_accum_steps      = GRAD_ACCUM_STEPS,
    num_epochs            = NUM_EPOCHS,
    lr_backbone           = LR * 0.1,
    lr_hnet               = LR,
    lr_lm                 = LR,
    weight_decay          = WEIGHT_DECAY,
    warmup_ratio          = WARMUP_RATIO,
    extra_params          = unfrozen_backbone_params + list(pos_head.parameters()),
    extra_params_lr       = 1e-7,
)

# ── Loss wrapper ──────────────────────────────────────────────────────────────
def hnet_loss_fn(logits, input_ids, attention_mask, p):
    return hnet_loss(
        logits, input_ids, attention_mask, p,
        target_ratio = TARGET_RATIO,
        lambda_ratio = LAMBDA_RATIO,
    )

# ── Training ──────────────────────────────────────────────────────────────────
loss_history      = init_loss_history()
best_loss         = float("inf")
global_step       = 0
optimizer_step    = 0
pos_step_count    = 0
SAVE_WARMUP_STEPS = 10

encoder.train()
lm_head.train()
pos_head.train()

print(f"\nStarting task-guided pretraining...")
print(f"  LAMBDA_POS      : {LAMBDA_POS}")
print(f"  LAMBDA_ALIGN    : {LAMBDA_ALIGN}")
print(f"  POS_BATCH_FREQ  : every {POS_BATCH_FREQ} AR steps")
print(f"  N_LOCAL_LAYERS  : {N_LOCAL_LAYERS}")
print(f"  N_UNFREEZE_LAYERS: {N_UNFREEZE_LAYERS}")

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()

    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):

        # ── Every POS_BATCH_FREQ steps, run one POS step ──────────────────
        if global_step % POS_BATCH_FREQ == 0:
            pos_batch   = next(pos_iter)
            pos_metrics = pos_step(
                pos_batch        = pos_batch,
                encoder          = encoder,
                pos_head         = pos_head,
                optimizer        = optimizer,
                scheduler        = scheduler,
                global_step      = global_step,
                grad_accum_steps = GRAD_ACCUM_STEPS,
                max_grad_norm    = MAX_GRAD_NORM,
                device           = DEVICE,
                lambda_pos       = LAMBDA_POS,
            )
            pos_step_count += 1
            if pos_step_count % 100 == 0:
                print(f"  [pos step {pos_step_count}] "
                      f"loss={pos_metrics['loss_pos'].item():.4f}  "
                      f"acc={pos_metrics['pos_acc']:.4f}")

        # ── AR pretraining step ───────────────────────────────────────────
        metrics = pretrain_step(
            batch              = batch,
            encoder            = encoder,
            lm_head            = lm_head,
            hnet_loss_fn       = hnet_loss_fn,
            optimizer          = optimizer,
            scheduler          = scheduler,
            global_step        = global_step,
            grad_accum_steps   = GRAD_ACCUM_STEPS,
            max_grad_norm      = MAX_GRAD_NORM,
            device             = DEVICE,
            lambda_align       = LAMBDA_ALIGN,
            byte_align_targets = byte_align_targets,
        )

        if metrics["did_step"]:
            optimizer_step += 1

            update_loss_history(
                loss_history,
                step        = optimizer_step,
                total       = metrics["total_loss"].item(),
                ar          = metrics["loss_ar"].item(),
                ratio       = metrics["loss_ratio"].item(),
                align       = metrics["loss_align"].item(),
                bpb         = metrics["bpb"].item(),
                boundary    = metrics["avg_boundary_ratio"].item(),
                p_stats     = metrics["p_stats"],
                chunk_stats = metrics["chunk_stats"],
            )

            if optimizer_step % 500 == 0:
                print(
                    f"  [step {optimizer_step:5d}] "
                    f"loss={metrics['total_loss'].item():.4f}  "
                    f"bpb={metrics['bpb'].item():.4f}  "
                    f"boundary={metrics['avg_boundary_ratio'].item():.3f}  "
                    f"align={metrics['loss_align'].item():.4f}"
                )
                save_checkpoint(
                    path        = os.path.join(PRETRAIN_SAVE_DIR, f"step_{optimizer_step}.pt"),
                    encoder     = encoder,
                    peft_model  = base,
                    lm_head     = lm_head,
                    optimizer   = optimizer,
                    scheduler   = scheduler,
                    global_step = optimizer_step,
                    best_loss   = best_loss,
                    config      = {
                        "model_id":          MODEL_ID,
                        "target_ratio":      TARGET_RATIO,
                        "lambda_ratio":      LAMBDA_RATIO,
                        "lambda_align":      LAMBDA_ALIGN,
                        "lambda_pos":        LAMBDA_POS,
                        "pos_batch_freq":    POS_BATCH_FREQ,
                        "n_local_layers":    N_LOCAL_LAYERS,
                        "n_unfreeze_layers": N_UNFREEZE_LAYERS,
                        "hidden_size":       HIDDEN_SIZE,
                    },
                )
                save_loss_history(loss_history, PRETRAIN_SAVE_DIR)
                print(f"  ✓ Saved step_{optimizer_step}.pt")

            if (optimizer_step >= SAVE_WARMUP_STEPS and
                    metrics["total_loss"].item() > 0 and
                    metrics["total_loss"].item() < best_loss):
                best_loss = metrics["total_loss"].item()
                save_checkpoint(
                    path        = os.path.join(PRETRAIN_SAVE_DIR, "best.pt"),
                    encoder     = encoder,
                    peft_model  = base,
                    lm_head     = lm_head,
                    optimizer   = optimizer,
                    scheduler   = scheduler,
                    global_step = optimizer_step,
                    best_loss   = best_loss,
                    config      = {
                        "model_id":          MODEL_ID,
                        "target_ratio":      TARGET_RATIO,
                        "lambda_ratio":      LAMBDA_RATIO,
                        "lambda_align":      LAMBDA_ALIGN,
                        "lambda_pos":        LAMBDA_POS,
                        "pos_batch_freq":    POS_BATCH_FREQ,
                        "n_local_layers":    N_LOCAL_LAYERS,
                        "n_unfreeze_layers": N_UNFREEZE_LAYERS,
                        "hidden_size":       HIDDEN_SIZE,
                    },
                )
                print(f"  ✓ Saved best.pt (loss={best_loss:.4f})")

        global_step += 1

    print(f"\n── Epoch {epoch+1} complete ──")
    print(f"   optimizer_step = {optimizer_step}")
    print(f"   pos_steps      = {pos_step_count}")
    print(f"   best_loss      = {best_loss:.4f}")
    save_loss_history(loss_history, PRETRAIN_SAVE_DIR)

wait_for_checkpoint()
save_checkpoint(
    path        = os.path.join(PRETRAIN_SAVE_DIR, "final.pt"),
    encoder     = encoder,
    peft_model  = base,
    lm_head     = lm_head,
    optimizer   = optimizer,
    scheduler   = scheduler,
    global_step = optimizer_step,
    best_loss   = best_loss,
    config      = {
        "model_id":          MODEL_ID,
        "target_ratio":      TARGET_RATIO,
        "lambda_ratio":      LAMBDA_RATIO,
        "lambda_align":      LAMBDA_ALIGN,
        "lambda_pos":        LAMBDA_POS,
        "pos_batch_freq":    POS_BATCH_FREQ,
        "n_local_layers":    N_LOCAL_LAYERS,
        "n_unfreeze_layers": N_UNFREEZE_LAYERS,
        "hidden_size":       HIDDEN_SIZE,
    },
    blocking = True,
)
save_loss_history(loss_history, PRETRAIN_SAVE_DIR)
print(f"\nPretraining complete.")
print(f"  Best loss      : {best_loss:.4f}")
print(f"  Optimizer steps: {optimizer_step}")
print(f"  POS steps      : {pos_step_count}")
print(f"  Saved to       : {PRETRAIN_SAVE_DIR}/")