# scripts/eval_chunk_sentiment.py
"""
Chunk-level sentiment analysis evaluation for H-Net.

Protocol:
  ZS: Train probe on Hindi 80% → evaluate on ALL Marathi/Urdu
  XL: Train fresh probe on 80% target language → evaluate on 20%

Labels:   Positive, Negative  (None examples skipped)
Input:    INDIC REVIEW field from IndicSentiment JSON
Pooling:  mean pool chunk embeddings → linear classifier
Metric:   accuracy + macro F1

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_chunk_sentiment \
        --hnet_ckpt  runs_qwen/hnet_pretrain_pos_guided/step_7000.pt \
        --train_data data/indic_sentiment/hi_test.json \
        --test_langs \
            Marathi:data/indic_sentiment/mr_test.json \
            Urdu:data/indic_sentiment/ur_test.json \
        --save_dir   runs_qwen/hnet_chunk_sentiment \
        --epochs     20 --n_local_layers 1
"""

import os
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

from models.load_hnet import load_hnet_encoder
from configs.default import DEVICE, DTYPE, MODEL_ID


# ── Data reader ───────────────────────────────────────────────────────────────

def read_sentiment(path):
    """Read IndicSentiment JSONL file. Returns list of (text, label)."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d     = json.loads(line)
            text  = d.get("INDIC REVIEW", "").strip()
            label = d.get("LABEL", None)
            if label in ("Positive", "Negative") and text:
                data.append((text, label))
    return data


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChunkSentimentDataset(Dataset):
    """
    Encodes each review as UTF-8 bytes.
    Label is sentence-level (one per review).
    Chunk embeddings will be mean-pooled for classification.
    """
    PAD_ID = 0
    EOS_ID = 1

    def __init__(self, data, label2id, max_len=512,
                 split="train", train_ratio=0.8, seed=42):
        self.label2id = label2id
        self.max_len  = max_len

        rng  = random.Random(seed)
        d    = data[:]
        rng.shuffle(d)
        cut  = int(len(d) * train_ratio)
        self.data = d[:cut] if split == "train" else d[cut:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        byte_ids = list(text.encode("utf-8"))[:self.max_len - 1]
        byte_ids.append(self.EOS_ID)
        pad_len   = self.max_len - len(byte_ids)
        byte_ids += [self.PAD_ID] * pad_len

        return {
            "input_ids":      torch.tensor(byte_ids, dtype=torch.long),
            "attention_mask": (torch.tensor(byte_ids,
                               dtype=torch.long) != self.PAD_ID).long(),
            "label":          torch.tensor(
                               self.label2id[label], dtype=torch.long),
        }


# ── Sentiment head ────────────────────────────────────────────────────────────

class ChunkSentimentHead(nn.Module):
    def __init__(self, d_model, n_classes, dropout=0.1):
        super().__init__()
        hidden          = min(256, d_model // 4)
        self.dropout    = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(d_model, hidden)
        self.act        = nn.GELU()
        self.norm       = nn.LayerNorm(hidden)
        self.proj       = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.dropout(x)
        h = self.act(self.bottleneck(h))
        h = self.norm(h)
        return self.proj(h)


# ── Encoder forward ───────────────────────────────────────────────────────────

@torch.no_grad()
def get_sentence_emb(encoder, input_ids, attention_mask):
    """
    Run encoder and mean-pool chunk embeddings to get sentence vector.
    Returns: [B, D]
    """
    h_hat, p, aux = encoder(input_ids, attention_mask)
    chunk_out      = aux["chunk_out"]
    chunk_embs     = chunk_out.get("chunk_embs_smooth",
                                   chunk_out["chunk_embs"])
    # chunk_embs is List[Tensor[C_i, D]] — one per batch item
    # Mean pool each item
    sentence_embs = torch.stack(
        [emb.mean(dim=0) for emb in chunk_embs], dim=0
    )  # [B, D]
    return sentence_embs


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(encoder, head, loader, id2label):
    encoder.eval(); head.eval()
    all_true, all_pred = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        sent_embs = get_sentence_emb(encoder, input_ids, attention_mask)
        sent_embs = sent_embs.to(dtype=DTYPE)
        logits    = head(sent_embs)
        preds     = logits.argmax(-1)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(all_true, all_pred) * 100
    f1  = f1_score(all_true, all_pred, average="macro") * 100
    return acc, f1


# ── Train probe ───────────────────────────────────────────────────────────────

def train_probe(encoder, head, train_loader, val_loader,
                id2label, args, desc=""):
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )
    best_acc, best_f1 = 0.0, 0.0

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss, n_batches = 0.0, 0

        for batch in tqdm(train_loader,
                          desc=f"{desc} Epoch {epoch}/{args.epochs}",
                          leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            with torch.no_grad():
                sent_embs = get_sentence_emb(
                    encoder, input_ids, attention_mask)
            sent_embs = sent_embs.to(dtype=DTYPE)

            logits = head(sent_embs)
            loss   = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); n_batches += 1

        val_acc, val_f1 = evaluate(encoder, head, val_loader, id2label)
        print(f"  {desc} Epoch {epoch:3d}/{args.epochs} | "
              f"loss={total_loss/max(n_batches,1):.4f} | "
              f"val_acc={val_acc:.1f}%  val_F1={val_f1:.1f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1  = val_f1

    return best_acc, best_f1


# ── Argparse ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hnet_ckpt",      required=True)
    p.add_argument("--train_data",     required=True,
                   help="Hindi IndicSentiment JSON")
    p.add_argument("--test_langs",     nargs="+", default=[],
                   help="Name:path e.g. Marathi:data/indic_sentiment/mr_test.json")
    p.add_argument("--save_dir",       default="runs_qwen/hnet_chunk_sentiment")
    p.add_argument("--max_len",        type=int,   default=512)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--n_local_layers", type=int,   default=1)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Parse test_langs
    test_langs = {}
    for item in args.test_langs:
        name, path = item.split(":", 1)
        if os.path.exists(path):
            test_langs[name] = path
        else:
            print(f"  WARNING: {path} not found — skipping {name}")

    # ── Load encoder ──────────────────────────────────────────────────────
    print("Loading H-Net encoder...")
    encoder, _ = load_hnet_encoder(
        checkpoint_path = args.hnet_ckpt,
        model_id        = MODEL_ID,
        device          = str(DEVICE),
        dtype           = DTYPE,
        frozen          = True,
        load_lm_head    = False,
        n_local_layers  = args.n_local_layers,
    )
    encoder.eval()
    d_model = encoder.byte_embedding.embedding_dim
    print(f"  Encoder d_model: {d_model}")

    # ── Build label vocab ─────────────────────────────────────────────────
    label2id = {"Negative": 0, "Positive": 1}
    id2label = {0: "Negative", 1: "Positive"}
    n_classes = 2
    print(f"  Labels: {label2id}")

    # ── Load Hindi data ───────────────────────────────────────────────────
    print(f"\nLoading Hindi sentiment: {args.train_data}")
    hindi_data = read_sentiment(args.train_data)
    print(f"  Examples: {len(hindi_data)}")

    train_ds = ChunkSentimentDataset(
        hindi_data, label2id, args.max_len, "train", 0.8, 42)
    val_ds   = ChunkSentimentDataset(
        hindi_data, label2id, args.max_len, "val",   0.8, 42)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Train Hindi probe ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1: Train sentiment probe on Hindi")
    print(f"{'='*60}")

    head = ChunkSentimentHead(d_model, n_classes).to(DEVICE, dtype=DTYPE)
    print(f"  Head params: {sum(p.numel() for p in head.parameters()):,}")

    best_hindi_acc, best_hindi_f1 = train_probe(
        encoder, head, train_loader, val_loader, id2label, args, "Hindi")
    print(f"\n  Hindi best val acc: {best_hindi_acc:.1f}%  "
          f"F1: {best_hindi_f1:.1f}%")

    results = {"Hindi_XL_acc": best_hindi_acc,
               "Hindi_XL_f1":  best_hindi_f1}

    # ── Cross-lingual evaluation ───────────────────────────────────────────
    def eval_language(lang_name, lang_path, hindi_probe):
        print(f"\n{'='*60}")
        print(f"{lang_name.upper()} EVALUATION")
        print(f"{'='*60}")

        lang_data = read_sentiment(lang_path)
        print(f"  Total examples: {len(lang_data)}")

        # ── ZS: Hindi probe → all target data ─────────────────────────────
        print(f"\n  [ZS] Zero-shot: Hindi probe on all {lang_name} data")
        zs_ds = ChunkSentimentDataset(
            lang_data, label2id, args.max_len,
            split="val", train_ratio=0.0, seed=42)
        zs_loader = DataLoader(zs_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=0)
        zs_acc, zs_f1 = evaluate(encoder, hindi_probe, zs_loader, id2label)
        print(f"  [ZS] {lang_name} acc={zs_acc:.1f}%  F1={zs_f1:.1f}%")

        # ── XL: fresh probe on 80% target data ────────────────────────────
        print(f"\n  [XL] Cross-lingual: fresh probe on 80% {lang_name}")
        xl_train_ds = ChunkSentimentDataset(
            lang_data, label2id, args.max_len, "train", 0.8, 42)
        xl_val_ds   = ChunkSentimentDataset(
            lang_data, label2id, args.max_len, "val",   0.8, 42)
        xl_train_loader = DataLoader(xl_train_ds, batch_size=args.batch_size,
                                     shuffle=True,  num_workers=0)
        xl_val_loader   = DataLoader(xl_val_ds,   batch_size=args.batch_size,
                                     shuffle=False, num_workers=0)
        print(f"  Train: {len(xl_train_ds)} | Val: {len(xl_val_ds)}")

        fresh_head = ChunkSentimentHead(
            d_model, n_classes).to(DEVICE, dtype=DTYPE)
        xl_acc, xl_f1 = train_probe(
            encoder, fresh_head, xl_train_loader, xl_val_loader,
            id2label, args, lang_name)
        print(f"  [XL] {lang_name} best acc={xl_acc:.1f}%  F1={xl_f1:.1f}%")

        return zs_acc, zs_f1, xl_acc, xl_f1

    for lang_name, lang_path in test_langs.items():
        zs_acc, zs_f1, xl_acc, xl_f1 = eval_language(
            lang_name, lang_path, head)
        results[f"{lang_name}_ZS_acc"] = zs_acc
        results[f"{lang_name}_ZS_f1"]  = zs_f1
        results[f"{lang_name}_XL_acc"] = xl_acc
        results[f"{lang_name}_XL_f1"]  = xl_f1

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — H-Net Chunk Sentiment (Acc / Macro-F1)")
    print(f"{'='*60}")
    print(f"  ZS = zero-shot  |  XL = cross-lingual (80% target data)")
    print(f"  Random baseline = 50.0% acc")
    print()
    print(f"{'Language':12s} | {'ZS Acc':8s} | {'ZS F1':8s} | "
          f"{'XL Acc':8s} | {'XL F1':8s}")
    print("-"*55)
    print(f"{'Hindi':12s} | {'---':8s} | {'---':8s} | "
          f"{best_hindi_acc:6.1f}%  | {best_hindi_f1:6.1f}%")
    for lang_name in test_langs:
        zs_a = results.get(f"{lang_name}_ZS_acc", 0)
        zs_f = results.get(f"{lang_name}_ZS_f1",  0)
        xl_a = results.get(f"{lang_name}_XL_acc", 0)
        xl_f = results.get(f"{lang_name}_XL_f1",  0)
        print(f"{lang_name:12s} | {zs_a:6.1f}%  | {zs_f:6.1f}%  | "
              f"{xl_a:6.1f}%  | {xl_f:6.1f}%")

    torch.save(results, os.path.join(args.save_dir, "results.pt"))
    print(f"\nSaved to: {args.save_dir}/results.pt")


if __name__ == "__main__":
    main()