# tasks/train_sentiment.py
"""
Sentiment Analysis fine-tuning on top of pretrained H-Net.

Encodes the full input text, pools byte-level representations,
and classifies into sentiment classes.

Usage:
    python tasks/train_sentiment.py \
        --hnet_ckpt runs/hnet_pretrain/best.pt \
        --data      path/to/sentiment.csv \
        --text_col  text \
        --label_col sentiment \
        --save_dir  runs/hnet_sentiment \
        --pool      mean
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import get_cosine_schedule_with_warmup

from models.load_hnet import load_hnet_encoder
from tasks.task_heads import SentimentHead
from data.sentiment_dataset import get_sentiment_dataloaders
from configs.default import DEVICE, DTYPE, SENTIMENT_NUM_EPOCHS, SENTIMENT_LR, MODEL_ID

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hnet_ckpt",  required=True)
    p.add_argument("--data",       required=True)
    p.add_argument("--text_col",   default="text")
    p.add_argument("--label_col",  default="label")
    p.add_argument("--sheet",      default=None)
    p.add_argument("--max_len",    type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs",     type=int, default=SENTIMENT_NUM_EPOCHS)
    p.add_argument("--lr",         type=float, default=SENTIMENT_LR)
    p.add_argument("--pool",       choices=["mean", "last", "cls"], default="mean")
    p.add_argument("--save_dir",   default="runs/hnet_sentiment")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, label2id, n_classes = \
        get_sentiment_dataloaders(
            data_path   = args.data,
            text_col    = args.text_col,
            label_col   = args.label_col,
            sheet_name  = args.sheet,
            max_len     = args.max_len,
            batch_size  = args.batch_size,
        )
    print(f"Classes ({n_classes}): {label2id}")
    torch.save(label2id, os.path.join(args.save_dir, "label2id.pt"))

    # ── 2. Load encoder (frozen) ──────────────────────────────────────────
    print("Loading pretrained H-Net encoder...")
    encoder, _ = load_hnet_encoder(
        checkpoint_path = args.hnet_ckpt,
        model_id        = MODEL_ID,
        device          = str(DEVICE),
        dtype           = DTYPE,
        frozen          = True,
        load_lm_head    = False,
    )
    encoder.eval()

    # ── 3. Sentiment head ─────────────────────────────────────────────────
    d_model = encoder.byte_embedding.embedding_dim
    sent_head = SentimentHead(
        d_model   = d_model,
        n_classes = n_classes,
        pool      = args.pool,
        dropout   = 0.1,
    ).to(DEVICE, dtype=DTYPE)

    # ── 4. Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        sent_head.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )

    # ── 5. Training loop ──────────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        sent_head.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            with torch.no_grad():
                h_hat, _, _ = encoder(input_ids, attention_mask)

            loss = sent_head.compute_loss(h_hat, attention_mask, labels)
            acc  = sent_head.accuracy(h_hat, attention_mask, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sent_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_acc  += acc
            n_batches  += 1

        train_loss /= n_batches
        train_acc  /= n_batches

        # ── Validation ────────────────────────────────────────────────────
        val_loss, val_acc = _eval_sentiment(encoder, sent_head, val_loader, DEVICE)
        print(
            f"Epoch {epoch+1} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "sent_head": sent_head.state_dict(),
                    "label2id":  label2id,
                    "n_classes": n_classes,
                    "d_model":   d_model,
                    "pool":      args.pool,
                },
                os.path.join(args.save_dir, "best_sent_head.pt"),
            )
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

    # ── 6. Final test evaluation ──────────────────────────────────────────
    print("\n── Test Set Evaluation ──")
    ckpt = torch.load(os.path.join(args.save_dir, "best_sent_head.pt"))
    sent_head.load_state_dict(ckpt["sent_head"])

    all_preds, all_labels = _collect_predictions(
        encoder, sent_head, test_loader, DEVICE
    )
    id2label = {v: k for k, v in label2id.items()}
    target_names = [id2label[i] for i in range(n_classes)]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print(f"Sentiment fine-tuning complete. Best val accuracy: {best_val_acc:.4f}")


@torch.no_grad()
def _eval_sentiment(encoder, sent_head, loader, device):
    encoder.eval()
    sent_head.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        h_hat, _, _ = encoder(input_ids, attention_mask)
        loss = sent_head.compute_loss(h_hat, attention_mask, labels)
        acc  = sent_head.accuracy(h_hat, attention_mask, labels)

        total_loss += loss.item()
        total_acc  += acc
        n          += 1

    return total_loss / max(n, 1), total_acc / max(n, 1)


@torch.no_grad()
def _collect_predictions(encoder, sent_head, loader, device):
    encoder.eval()
    sent_head.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        h_hat, _, _ = encoder(input_ids, attention_mask)
        preds = sent_head.predict(h_hat, attention_mask)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


if __name__ == "__main__":
    main()