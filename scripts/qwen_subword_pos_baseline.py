# scripts/qwen_subword_pos_baseline.py
"""
Subword baseline for POS tagging using Qwen2.5-1.5B.

Protocol (consistent with eval_chunk_pos.py):
  1. Train a linear POS probe on top of FROZEN Qwen subword representations
     using Hindi UD treebank (80% train / 20% val split)
  2. For cross-lingual transfer (Bhojpuri, Marathi):
     - Train a NEW probe on 80% of target language data
     - Evaluate on remaining 20%
     - Encoder stays FROZEN throughout
     - Same 80/20 split and seed=42 as eval_chunk_pos.py

This ensures a fair comparison with H-Net chunk-level results.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m scripts.qwen_subword_pos_baseline \
        --train_data data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
        --test_hindi data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
        --test_bho   data/ud_bhojpuri/bho_bhtb-ud-test.conllu \
        --test_mr    data/ud_marathi/mr_ufal-ud-train.conllu \
        --save_dir   runs_qwen/qwen_subword_pos_fixed \
        --epochs     20
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup
from configs.default import MODEL_ID, DEVICE, DTYPE


# ── CoNLL-U reader ─────────────────────────────────────────────────────────────

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


# ── Subword POS Dataset ────────────────────────────────────────────────────────

class SubwordPOSDataset(Dataset):
    """
    Tokenizes each sentence with Qwen tokenizer.
    Labels assigned to FIRST subword token of each word.
    Other subword tokens get IGNORE_IDX = -100.
    """
    IGNORE_IDX = -100

    def __init__(self, sentences, tag2id, tokenizer, max_len=512,
                 split="train", train_ratio=0.8, seed=42):
        self.tag2id    = tag2id
        self.tokenizer = tokenizer
        self.max_len   = max_len

        import random
        rng  = random.Random(seed)
        data = sentences[:]
        rng.shuffle(data)
        cut  = int(len(data) * train_ratio)
        self.sentences = data[:cut] if split == "train" else data[cut:]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words, tags = self.sentences[idx]

        input_ids = [self.tokenizer.bos_token_id or 1]
        label_ids = [self.IGNORE_IDX]

        for word, tag in zip(words, tags):
            toks = self.tokenizer.encode(" " + word, add_special_tokens=False)
            if not toks:
                continue
            tag_id = self.tag2id.get(tag, 0)
            label_ids.append(tag_id)
            input_ids.append(toks[0])
            for t in toks[1:]:
                input_ids.append(t)
                label_ids.append(self.IGNORE_IDX)

        input_ids.append(self.tokenizer.eos_token_id or 2)
        label_ids.append(self.IGNORE_IDX)

        input_ids = input_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]

        pad_len    = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id or 0] * pad_len
        label_ids += [self.IGNORE_IDX] * pad_len

        attn_mask = [1 if i != (self.tokenizer.pad_token_id or 0)
                     else 0 for i in input_ids]

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
        }


# ── Linear POS head ────────────────────────────────────────────────────────────

class SubwordPOSHead(nn.Module):
    def __init__(self, hidden_size, n_tags, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, n_tags)

    def forward(self, x):
        return self.proj(self.drop(x))

    def compute_loss(self, logits, labels):
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def accuracy(self, logits, labels):
        preds  = logits.argmax(-1).reshape(-1)
        labels = labels.reshape(-1)
        mask   = labels != -100
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == labels[mask]).float().mean().item()


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, pos_head, loader):
    model.eval()
    pos_head.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        encoder = getattr(model, "language_model", model)
        out    = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state.to(dtype=torch.float32)
        logits = pos_head(hidden)
        loss   = pos_head.compute_loss(logits, labels)
        acc    = pos_head.accuracy(logits, labels)
        total_loss += loss.item()
        total_acc  += acc
        n          += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


# ── Train probe helper ─────────────────────────────────────────────────────────

def train_probe(model, pos_head, train_loader, val_loader, args, desc=""):
    """
    Train a pos_head probe on top of frozen model.
    Returns best val accuracy.
    """
    optimizer = torch.optim.AdamW(
        pos_head.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 20)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        pos_head.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0

        for batch in tqdm(train_loader,
                          desc=f"{desc} Epoch {epoch}/{args.epochs}",
                          leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            with torch.no_grad():
                encoder = getattr(model, "language_model", model)
                out    = encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
                hidden = out.last_hidden_state.to(dtype=torch.float32)

            logits = pos_head(hidden)
            loss   = pos_head.compute_loss(logits, labels)
            acc    = pos_head.accuracy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pos_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_acc  += acc
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        train_acc  /= max(n_batches, 1)
        val_loss, val_acc = evaluate(model, pos_head, val_loader)

        print(f"  {desc} Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


# ── Evaluate one language 

def run_language(lang_name, sentences, tag2id, tokenizer,
                 model, hidden_size, n_tags, args):
    """
    Train probe on 80% of sentences, evaluate on 20%.
    Same protocol as eval_chunk_pos.py.
    Returns best val accuracy.
    """
    print(f"\nEvaluating: {lang_name} ({len(sentences)} sentences)")
    print(f"  Train: {int(len(sentences)*0.8)} | Val: {len(sentences)-int(len(sentences)*0.8)}")

    train_ds = SubwordPOSDataset(
        sentences, tag2id, tokenizer, args.max_len,
        split="train", train_ratio=0.8, seed=42
    )
    val_ds = SubwordPOSDataset(
        sentences, tag2id, tokenizer, args.max_len,
        split="val", train_ratio=0.8, seed=42
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Fresh probe for each language — same init
    pos_head = SubwordPOSHead(hidden_size, n_tags).to(DEVICE)

    best_acc = train_probe(
        model, pos_head, train_loader, val_loader, args,
        desc=lang_name
    )

    print(f"  {lang_name} best val accuracy: {best_acc*100:.1f}%")
    return best_acc


# ── Argparse 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", required=True,
                   help="Hindi UD CoNLL-U for training the Hindi probe")
    p.add_argument("--test_hindi", required=True,
                   help="Hindi UD CoNLL-U for Hindi evaluation")
    p.add_argument("--test_bho",   default=None,
                   help="Bhojpuri CoNLL-U")
    p.add_argument("--test_mr",    default=None,
                   help="Marathi CoNLL-U")
    p.add_argument("--save_dir",   default="runs_qwen/qwen_subword_pos_fixed")
    p.add_argument("--max_len",    type=int,   default=512)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=2e-4)
    return p.parse_args()


# ── Main 

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load frozen Qwen 
    print(f"Loading Qwen: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    model = model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    print(f"  Hidden size : {hidden_size}")
    print(f"  All params frozen")

    # ── Build tag vocab from Hindi ─────────────────────────────────────────
    print("\nBuilding tag vocabulary from Hindi data...")
    hindi_sentences = read_conll(args.train_data)
    all_tags = sorted({t for _, tags in hindi_sentences for t in tags})
    tag2id   = {t: i for i, t in enumerate(all_tags)}
    n_tags   = len(tag2id)
    print(f"  Total sentences : {len(hindi_sentences)}")
    print(f"  Tags ({n_tags})  : {all_tags}")
    torch.save(tag2id, os.path.join(args.save_dir, "tag2id.pt"))

    results = {}

    # ── Hindi ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("HINDI (in-language, 80/20 split)")
    print("="*60)
    results["Hindi"] = run_language(
        "Hindi", read_conll(args.test_hindi),
        tag2id, tokenizer, model, hidden_size, n_tags, args
    )

    # ── Bhojpuri ──────────────────────────────────────────────────────────
    if args.test_bho and os.path.exists(args.test_bho):
        print("\n" + "="*60)
        print("BHOJPURI (cross-lingual transfer, 80/20 split)")
        print("="*60)
        results["Bhojpuri"] = run_language(
            "Bhojpuri", read_conll(args.test_bho),
            tag2id, tokenizer, model, hidden_size, n_tags, args
        )

    # ── Marathi ───────────────────────────────────────────────────────────
    if args.test_mr and os.path.exists(args.test_mr):
        print("\n" + "="*60)
        print("MARATHI (cross-lingual transfer, 80/20 split)")
        print("="*60)
        results["Marathi"] = run_language(
            "Marathi", read_conll(args.test_mr),
            tag2id, tokenizer, model, hidden_size, n_tags, args
        )

    # ── Final summary ─────────────────────────────────────────────────────
    hnet = {"Hindi": 49.5, "Bhojpuri": 56.9, "Marathi": 42.1}
    sizes = {"Hindi": "13,306", "Bhojpuri": "357", "Marathi": "~500"}

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Protocol: 80/20 train/val split, seed=42 (same as H-Net)")
    print(f"Encoder: ")
    print(f"Probe:    fresh probe trained per language")
    print()
    print(f"{'Language':12s} | {'Sentences':10s} | {'Qwen subword':12s} | "
          f"{'H-Net chunk':12s} | {'Δ':8s}")
    print("-"*65)
    for lang, acc in results.items():
        q   = acc * 100
        h   = hnet.get(lang, 0.0)
        gap = q - h
        print(f"{lang:12s} | {sizes.get(lang,'?'):10s} | {q:8.1f}%     | "
              f"{h:8.1f}%     | {gap:+.1f} pp")

    torch.save(results, os.path.join(args.save_dir, "results.pt"))
    print(f"\nResults saved to: {args.save_dir}/results.pt")


if __name__ == "__main__":
    main()