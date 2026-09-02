# scripts/eval_cpt_sentiment.py
"""
Evaluate CPT baseline on Sentiment Analysis.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m scripts.eval_cpt_sentiment \
        --cpt_ckpt   runs_cpt/Qwen_Qwen3-4B/best.pt \
        --train_data data/indic_sentiment/hi_test.json \
        --test_langs \
            Marathi:data/indic_sentiment/mr_test.json \
            Urdu:data/indic_sentiment/ur_test.json \
        --save_dir   runs_cpt/sentiment \
        --epochs     20
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score

from configs.default import DEVICE, DTYPE, MODEL_ID


# ── Dataset ───────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, data, label2id, tokenizer, max_len=256,
                 split="train", train_ratio=0.8, seed=42):
        self.tokenizer = tokenizer
        self.label2id  = label2id
        self.max_len   = max_len

        import random
        rng  = random.Random(seed)
        d    = data[:]
        rng.shuffle(d)
        cut  = int(len(d) * train_ratio)
        self.data = d[:cut] if split == "train" else d[cut:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        enc = self.tokenizer(
            text,
            max_length     = self.max_len,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(
                self.label2id.get(label, 0), dtype=torch.long),
        }


def load_sentiment_data(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                text  = item.get("INDIC REVIEW", item.get("sentence", item.get("text", "")))
                label = item.get("LABEL", item.get("label", item.get("sentiment", "")))
                if text and label:
                    data.append((text, str(label)))
            except: continue
    return data


# ── Sentiment Head ────────────────────────────────────────────────────────────
class SentimentHead(nn.Module):
    def __init__(self, hidden_size, n_labels, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(hidden_size, n_labels)

    def forward(self, x):
        # Mean pooling over non-padding tokens
        return self.proj(self.dropout(x.mean(dim=1)))


# ── Load CPT model ────────────────────────────────────────────────────────────
def load_cpt_model(cpt_ckpt_path):
    ckpt     = torch.load(cpt_ckpt_path, map_location="cpu")
    config   = ckpt.get("config", {})
    model_id = config.get("model_id", MODEL_ID)

    print(f"  Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map="auto")
    model.load_state_dict(ckpt["model_state"], strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, config


# ── Evaluate accuracy ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, head, loader):
    model.eval(); head.eval()
    all_true, all_pred = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)
        # Mean pool over valid tokens
        hidden = outputs.hidden_states[-1]
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)

        logits = head.proj(head.dropout(pooled.to(head.proj.weight.dtype)))
        preds  = logits.argmax(-1)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    return accuracy_score(all_true, all_pred) * 100


# ── Train probe ───────────────────────────────────────────────────────────────
def train_probe(model, head, train_loader, val_loader, args):
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        head.train()
        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch}/{args.epochs}",
                          leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                mask   = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)

            logits = head.proj(head.dropout(pooled.to(head.proj.weight.dtype)))
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        val_acc = evaluate(model, head, val_loader)
        print(f"  Epoch {epoch}/{args.epochs}  val_acc={val_acc:.1f}%")
        if val_acc > best_acc:
            best_acc = val_acc
    return best_acc


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_ckpt",   required=True)
    p.add_argument("--train_data", required=True)
    p.add_argument("--test_langs", nargs="+", required=True)
    p.add_argument("--save_dir",   default="runs_cpt/sentiment")
    p.add_argument("--max_len",    type=int,   default=256)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=2e-4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading CPT model...")
    model, config = load_cpt_model(args.cpt_ckpt)
    tokenizer   = AutoTokenizer.from_pretrained(
        config.get("model_id", MODEL_ID))
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "text_config", model.config).hidden_size

    print("Loading Hindi sentiment data...")
    hi_data    = load_sentiment_data(args.train_data)
    all_labels = sorted({l for _, l in hi_data})
    label2id   = {l: i for i, l in enumerate(all_labels)}
    n_labels   = len(label2id)
    print(f"  Examples: {len(hi_data)}  Labels: {all_labels}")

    train_ds = SentimentDataset(hi_data, label2id, tokenizer,
                                 args.max_len, "train")
    val_ds   = SentimentDataset(hi_data, label2id, tokenizer,
                                 args.max_len, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print("\nTraining Hindi sentiment probe...")
    head      = SentimentHead(hidden_size, n_labels).to(DEVICE, dtype=DTYPE)
    hindi_acc = train_probe(model, head, train_loader, val_loader, args)
    print(f"Hindi XL accuracy: {hindi_acc:.1f}%")

    results = {"Hindi_XL": hindi_acc}

    for lang_spec in args.test_langs:
        lang, path = lang_spec.split(":", 1)
        print(f"\n{'='*50}")
        print(f"Evaluating: {lang}")

        lang_data = load_sentiment_data(path)
        print(f"  Examples: {len(lang_data)}")

        # ZS
        zs_ds = SentimentDataset(lang_data, label2id, tokenizer,
                                  args.max_len, "val", train_ratio=0.0)
        zs_loader = DataLoader(zs_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=0)
        zs_acc = evaluate(model, head, zs_loader)
        print(f"  ZS accuracy: {zs_acc:.1f}%")

        # XL
        xl_train_ds = SentimentDataset(lang_data, label2id, tokenizer,
                                        args.max_len, "train")
        xl_val_ds   = SentimentDataset(lang_data, label2id, tokenizer,
                                        args.max_len, "val")
        xl_train_loader = DataLoader(xl_train_ds, batch_size=args.batch_size,
                                     shuffle=True,  num_workers=0)
        xl_val_loader   = DataLoader(xl_val_ds,   batch_size=args.batch_size,
                                     shuffle=False, num_workers=0)
        fresh_head = SentimentHead(hidden_size, n_labels).to(
            DEVICE, dtype=DTYPE)
        xl_acc = train_probe(model, fresh_head, xl_train_loader,
                             xl_val_loader, args)
        print(f"  XL accuracy: {xl_acc:.1f}%")

        results[f"{lang}_ZS"] = zs_acc
        results[f"{lang}_XL"] = xl_acc

    print(f"\n{'='*50}")
    print("FINAL SUMMARY — CPT Sentiment Baseline")
    for k, v in results.items():
        print(f"  {k}: {v:.1f}%")

    torch.save(results, os.path.join(args.save_dir, "results.pt"))
    print(f"\nSaved to: {args.save_dir}/results.pt")


if __name__ == "__main__":
    main()