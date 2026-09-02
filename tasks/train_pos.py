# tasks/train_pos.py
"""
POS Tagging fine-tuning on top of pretrained H-Net.

Two modes:
  "frozen"  — H-Net encoder fully frozen; only POSHead trained.
  "partial" — Unfreeze routing + smoother + skip_proj + local enc/dec.

Usage:
    python -m tasks.train_pos \
        --hnet_ckpt runs_v2/hnet_pretrain/best.pt \
        --data      path/to/ud_hindi.conll \
        --save_dir  runs/hnet_pos \
        --mode      frozen
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from models.load_hnet import load_hnet_encoder
from tasks.task_heads import POSHead
from data.pos_dataset import get_pos_dataloaders
from configs.default  import DEVICE, DTYPE, POS_NUM_EPOCHS, POS_LR, MODEL_ID


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hnet_ckpt",      required=True)
    p.add_argument("--data",           required=True)
    p.add_argument("--max_len",        type=int,   default=512)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--epochs",         type=int,   default=POS_NUM_EPOCHS)
    p.add_argument("--lr",             type=float, default=POS_LR)
    p.add_argument("--mode",           choices=["frozen", "partial"], default="frozen")
    p.add_argument("--save_dir",       default="runs/hnet_pos")
    p.add_argument("--n_local_layers", type=int,   default=3)   # match pretraining config
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("Loading data...")
    train_loader, val_loader, tag2id, n_tags = get_pos_dataloaders(
        conll_path = args.data,
        max_len    = args.max_len,
        batch_size = args.batch_size,
    )
    print(f"  Tags ({n_tags}): {tag2id}")
    torch.save(tag2id, os.path.join(args.save_dir, "tag2id.pt"))

    # ── 2. Load encoder ───────────────────────────────────────────────────
    print("Loading pretrained H-Net encoder...")
    encoder, _ = load_hnet_encoder(
        checkpoint_path = args.hnet_ckpt,
        model_id        = MODEL_ID,
        device          = str(DEVICE),
        dtype           = DTYPE,
        frozen          = True,
        load_lm_head    = False,
        n_local_layers  = args.n_local_layers,   # ← pass through
    )

    if args.mode == "partial":
        for m in [encoder.routing, encoder.smoother, encoder.skip_proj,
                  encoder.local_encoder, encoder.local_decoder]:
            for param in m.parameters():
                param.requires_grad = True
        print("  Partial unfreeze: routing + smoother + skip_proj + local enc/dec")
    else:
        print("  Encoder fully frozen")

    # ── 3. POS head ───────────────────────────────────────────────────────
    d_model  = encoder.byte_embedding.embedding_dim
    pos_head = POSHead(d_model=d_model, n_tags=n_tags, dropout=0.1)
    pos_head = pos_head.to(DEVICE, dtype=DTYPE)

    # ── 4. Optimizer ──────────────────────────────────────────────────────
    trainable_params = list(pos_head.parameters())
    if args.mode == "partial":
        for m in [encoder.routing, encoder.smoother, encoder.skip_proj,
                  encoder.local_encoder, encoder.local_decoder]:
            trainable_params += list(m.parameters())

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Trainable params: {n_trainable:,}")

    optimizer   = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )

    # ── 5. Training loop ──────────────────────────────────────────────────
    best_val_acc = 0.0
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        pos_head.train()
        if args.mode == "partial":
            encoder.train()
        else:
            encoder.eval()

        train_loss, train_acc, n_batches = 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            if args.mode == "frozen":
                with torch.no_grad():
                    h_hat, _, _ = encoder(input_ids, attention_mask)
            else:
                h_hat, _, _ = encoder(input_ids, attention_mask)

            loss = pos_head.compute_loss(h_hat, labels)
            acc  = pos_head.word_accuracy(h_hat, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_acc  += acc
            n_batches  += 1

        train_loss /= n_batches
        train_acc  /= n_batches

        val_loss, val_acc = _eval_pos(encoder, pos_head, val_loader, DEVICE)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
        })
        torch.save(loss_history, os.path.join(args.save_dir, "loss_history.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "pos_head": pos_head.state_dict(),
                    "tag2id":   tag2id,
                    "n_tags":   n_tags,
                    "d_model":  d_model,
                    "val_acc":  best_val_acc,
                    "epoch":    epoch,
                    "mode":     args.mode,
                },
                os.path.join(args.save_dir, "best_pos_head.pt"),
            )
            print(f"  ✓ Saved best_pos_head.pt (val_acc={best_val_acc:.4f})")

    print(f"\nPOS fine-tuning complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
    print(f"  Saved to          : {args.save_dir}/")


@torch.no_grad()
def _eval_pos(encoder, pos_head, loader, device):
    encoder.eval()
    pos_head.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        h_hat, _, _ = encoder(input_ids, attention_mask)
        loss = pos_head.compute_loss(h_hat, labels)
        acc  = pos_head.word_accuracy(h_hat, labels)
        total_loss += loss.item()
        total_acc  += acc
        n          += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


if __name__ == "__main__":
    main()