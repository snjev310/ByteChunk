# scripts/llama_pos_finetune.py
"""
Fine-tune LLaMA 3.1-8B + LoRA directly on Hindi POS tagging.
No H-Net involved. Pure subword baseline.

Compares:
  LLaMA subword LoRA fine-tuned  vs  H-Net byte-level frozen probe (32.2%)

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m scripts.llama_pos_finetune \
        --data      data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
        --save_dir  runs/llama_pos \
        --epochs    20
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType

from configs.default import DEVICE, DTYPE, MODEL_ID


# ── Dataset ───────────────────────────────────────────────────────────────────

class SubwordPOSDataset(Dataset):
    """
    Tokenise Hindi sentences with LLaMA tokenizer.
    First subword token of each word gets the POS label.
    All other tokens get IGNORE_IDX=-100.
    """
    IGNORE_IDX = -100

    def __init__(self, sentences, tokenizer, tag2id,
                 max_len=512, split="train", train_ratio=0.8, seed=42):
        self.tokenizer = tokenizer
        self.tag2id    = tag2id
        self.max_len   = max_len

        import random
        rng = random.Random(seed)
        rng.shuffle(sentences)
        cut = int(len(sentences) * train_ratio)
        self.sentences = sentences[:cut] if split == "train" else sentences[cut:]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words, tags = self.sentences[idx]
        input_ids, labels = [], []

        for word, tag in zip(words, tags):
            toks = self.tokenizer.encode(" " + word, add_special_tokens=False)
            if not toks:
                continue
            tag_id = self.tag2id.get(tag, 0)
            for j, tok in enumerate(toks):
                input_ids.append(tok)
                labels.append(tag_id if j == 0 else self.IGNORE_IDX)

        # BOS
        bos = self.tokenizer.bos_token_id or 1
        input_ids = [bos] + input_ids[:self.max_len - 1]
        labels    = [self.IGNORE_IDX] + labels[:self.max_len - 1]

        # Pad
        pad_id  = self.tokenizer.pad_token_id or 0
        pad_len = self.max_len - len(input_ids)
        input_ids += [pad_id]          * pad_len
        labels    += [self.IGNORE_IDX] * pad_len

        ids  = torch.tensor(input_ids, dtype=torch.long)
        mask = (ids != pad_id).long()
        lbl  = torch.tensor(labels,    dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": lbl}


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── POS Head ─────────────────────────────────────────────────────────────────

class POSHead(nn.Module):
    """Same bottleneck architecture as H-Net POSHead for fair comparison."""
    def __init__(self, d_model, n_tags, dropout=0.1):
        super().__init__()
        hidden          = min(256, d_model // 4)
        self.dropout    = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(d_model, hidden)
        self.act        = nn.GELU()
        self.norm       = nn.LayerNorm(hidden)
        self.proj       = nn.Linear(hidden, n_tags)

    def forward(self, h):
        return self.proj(self.norm(self.act(self.bottleneck(self.dropout(h)))))

    def loss(self, h, labels):
        return F.cross_entropy(
            self.forward(h).reshape(-1, self.proj.out_features),
            labels.reshape(-1), ignore_index=-100,
        )

    @torch.no_grad()
    def accuracy(self, h, labels):
        preds   = self.forward(h).argmax(-1).reshape(-1)
        targets = labels.reshape(-1)
        mask    = targets != -100
        if mask.sum() == 0: return 0.0
        return (preds[mask] == targets[mask]).float().mean().item()


# ── CoNLL-U Reader ────────────────────────────────────────────────────────────

def read_conll(path):
    sentences, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "" or line.startswith("#"):
                if words:
                    sentences.append((words[:], tags[:]))
                    words, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) < 4: continue
            if "-" in parts[0] or "." in parts[0]: continue
            if parts[3] == "_": continue
            words.append(parts[1])
            tags.append(parts[3])
    if words:
        sentences.append((words, tags))
    return sentences


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--save_dir",   default="runs/llama_pos")
    p.add_argument("--max_len",    type=int,   default=512)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--lora_r",     type=int,   default=64)
    p.add_argument("--lora_alpha", type=int,   default=128)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading CoNLL-U data...")
    sentences = read_conll(args.data)
    all_tags  = sorted({t for _, tags in sentences for t in tags})
    tag2id    = {t: i for i, t in enumerate(all_tags)}
    n_tags    = len(tag2id)
    print(f"  Sentences: {len(sentences)}  |  Tags ({n_tags}): {all_tags}")

    # ── LLaMA + LoRA (fresh — no H-Net checkpoint) ────────────────────────
    print(f"\nLoading LLaMA: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map="auto",
    )
    lora_cfg = LoraConfig(
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        target_modules = ["q_proj", "k_proj", "v_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    hidden_size = model.config.hidden_size

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = SubwordPOSDataset(sentences, tokenizer, tag2id,
                                  args.max_len, "train")
    val_ds   = SubwordPOSDataset(sentences, tokenizer, tag2id,
                                  args.max_len, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── POS head ──────────────────────────────────────────────────────────
    pos_head = POSHead(hidden_size, n_tags).to(DEVICE, dtype=DTYPE)
    print(f"  POS head params: {sum(p.numel() for p in pos_head.parameters()):,}")

    # ── Optimizer — train LoRA + POS head together ─────────────────────────
    trainable = (
        [p for p in model.parameters() if p.requires_grad]
        + list(pos_head.parameters())
    )
    total_trainable = sum(p.numel() for p in trainable)
    print(f"  Total trainable: {total_trainable:,}")

    optimizer   = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    history      = []

    print(f"\nFine-tuning LLaMA + LoRA on Hindi POS for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pos_head.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            out = model.model(
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                output_hidden_states = True,
                return_dict          = True,
                use_cache            = False,
            )
            h    = out.hidden_states[-1]   # [B, T, 4096]
            loss = pos_head.loss(h, labels)
            acc  = pos_head.accuracy(h, labels)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_acc  += acc
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        train_acc  /= max(n_batches, 1)

        val_loss, val_acc = _eval(model, pos_head, val_loader)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
        })
        torch.save(history, os.path.join(args.save_dir, "loss_history.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"pos_head": pos_head.state_dict(),
                 "lora":     {k: v for k, v in model.named_parameters()
                              if "lora" in k and v.requires_grad},
                 "tag2id":   tag2id, "n_tags": n_tags,
                 "val_acc":  best_val_acc, "epoch": epoch},
                os.path.join(args.save_dir, "best.pt"),
            )
            print(f"  ✓ Saved best (val_acc={best_val_acc:.4f})")

    # ── Final comparison ──────────────────────────────────────────────────
    print("\n" + "="*58)
    print("  SUBWORD vs BYTE-LEVEL POS TAGGING COMPARISON")
    print("="*58)
    print(f"  Random baseline                    :  6.25%")
    print(f"  H-Net byte-level  (frozen probe)   : 32.2%")
    print(f"  LLaMA subword LoRA fine-tuned       : {best_val_acc*100:.1f}%")
    gap = (best_val_acc - 0.322) * 100
    print(f"  Subword advantage over byte-level  : {gap:+.1f} pp")
    print("="*58)
    print(f"Saved to: {args.save_dir}/")


@torch.no_grad()
def _eval(model, pos_head, loader):
    model.eval()
    pos_head.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        out  = model.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True, use_cache=False,
        )
        h    = out.hidden_states[-1]
        loss = pos_head.loss(h, labels)
        acc  = pos_head.accuracy(h, labels)
        if not torch.isnan(loss):
            total_loss += loss.item()
        total_acc += acc
        n         += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


if __name__ == "__main__":
    main()