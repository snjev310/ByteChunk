# scripts/qwen_zeroshot_eval.py
"""
Zero-shot Qwen subword POS evaluation.

Protocol:
  1. Train probe on Hindi (80/20 split, seed=42)
  2. Apply the SAME Hindi-trained probe directly to target languages
     with NO retraining (true zero-shot)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.qwen_zeroshot_eval \
        --train_data data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
        --test_langs \
            Sanskrit:data/ud_sanskrit/sa_ufal-ud-test.conllu \
            Magahi:data/ud_magahi/mag_mgtb-ud-test.conllu \
            Urdu:data/ud_urdu/ur_udtb-ud-test.conllu \
        --save_dir runs_qwen/qwen_zeroshot \
        --epochs   20
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class SubwordPOSDataset(Dataset):
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
            if not toks: continue
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
        attn_mask  = [1 if i != (self.tokenizer.pad_token_id or 0)
                      else 0 for i in input_ids]

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
        }


# ── POS head ──────────────────────────────────────────────────────────────────

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
            labels.reshape(-1), ignore_index=-100)

    def accuracy(self, logits, labels):
        preds  = logits.argmax(-1).reshape(-1)
        labels = labels.reshape(-1)
        mask   = labels != -100
        if mask.sum() == 0: return 0.0
        return (preds[mask] == labels[mask]).float().mean().item()


# ── Evaluate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, pos_head, loader):
    model.eval(); pos_head.eval()
    total_acc, n = 0.0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        encoder = getattr(model, "language_model", model)
        out    = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state.to(dtype=torch.float32)
        logits = pos_head(hidden)
        total_acc += pos_head.accuracy(logits, labels)
        n += 1
    return total_acc / max(n, 1)


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data",  required=True,
                   help="Hindi CoNLL-U for training the probe")
    p.add_argument("--test_langs",  nargs="+", required=True,
                   help="List of Name:path pairs e.g. Sanskrit:data/ud_sanskrit/sa.conllu")
    p.add_argument("--save_dir",    default="runs_qwen/qwen_zeroshot")
    p.add_argument("--max_len",     type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=2e-4)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Parse test_langs
    test_langs = {}
    for item in args.test_langs:
        name, path = item.split(":", 1)
        test_langs[name] = path
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {name}")

    # ── Load frozen Qwen ──────────────────────────────────────────────────
    print(f"Loading Qwen: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    hidden_size = (getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size)
    print(f"  Hidden size: {hidden_size}  |  All params frozen")

    # ── Build tag vocab from Hindi ─────────────────────────────────────────
    print(f"\nLoading Hindi training data: {args.train_data}")
    hindi_sents = read_conll(args.train_data)
    all_tags    = sorted({t for _, tags in hindi_sents for t in tags})
    tag2id      = {t: i for i, t in enumerate(all_tags)}
    n_tags      = len(tag2id)
    print(f"  Sentences: {len(hindi_sents)}  |  Tags ({n_tags}): {all_tags}")

    # ── Train probe on Hindi ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 1: Train probe on Hindi (80/20 split, seed=42)")
    print(f"{'='*60}")

    train_ds = SubwordPOSDataset(hindi_sents, tag2id, tokenizer,
                                 args.max_len, "train", 0.8, 42)
    val_ds   = SubwordPOSDataset(hindi_sents, tag2id, tokenizer,
                                 args.max_len, "val",   0.8, 42)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    pos_head  = SubwordPOSHead(hidden_size, n_tags).to(DEVICE)
    optimizer = torch.optim.AdamW(pos_head.parameters(),
                                  lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )

    best_hindi_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        pos_head.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"Hindi Epoch {epoch}/{args.epochs}",
                          leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            with torch.no_grad():
                encoder = getattr(model, "language_model", model)
                out    = encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state.to(dtype=torch.float32)
            logits = pos_head(hidden)
            loss   = pos_head.compute_loss(logits, labels)
            acc    = pos_head.accuracy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pos_head.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            train_loss += loss.item(); train_acc += acc; n_batches += 1

        val_acc = evaluate(model, pos_head, val_loader)
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train_acc={train_acc/max(n_batches,1):.4f} | "
              f"val_acc={val_acc:.4f}")
        if val_acc > best_hindi_acc:
            best_hindi_acc = val_acc
            torch.save(pos_head.state_dict(),
                       os.path.join(args.save_dir, "best_hindi_probe.pt"))

    print(f"\n  Hindi best val accuracy: {best_hindi_acc*100:.1f}%")

    # Load best Hindi probe
    pos_head.load_state_dict(
        torch.load(os.path.join(args.save_dir, "best_hindi_probe.pt"),
                   map_location=DEVICE)
    )

    # ── Zero-shot evaluation on target languages ───────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 2: Zero-shot evaluation (Hindi probe, NO retraining)")
    print(f"{'='*60}")
    print(f"  Encoder: frozen, pretrained on Hindi only")
    print(f"  Probe:   Hindi-trained, applied directly to target language")
    print()

    results = {"Hindi_probe": best_hindi_acc}

    for lang_name, lang_path in test_langs.items():
        if not os.path.exists(lang_path):
            continue

        sents = read_conll(lang_path)
        print(f"  {lang_name}: {len(sents)} sentences")

        # All data used as test — no target language training
        ds = SubwordPOSDataset(
            sents, tag2id, tokenizer, args.max_len,
            split="val", train_ratio=0.0, seed=42
        )
        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        acc = evaluate(model, pos_head, loader)
        results[f"{lang_name}_ZS"] = acc
        print(f"  {lang_name} ZS accuracy: {acc*100:.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT SUMMARY — Qwen Subword")
    print(f"{'='*60}")
    print(f"  Hindi probe accuracy: {best_hindi_acc*100:.1f}%")
    print()
    print(f"{'Language':12s} | {'ZS Accuracy':12s} | {'#Sentences':12s}")
    print("-"*42)
    sizes = {"Sanskrit": 230, "Magahi": 551, "Urdu": 535}
    for lang_name in test_langs:
        key = f"{lang_name}_ZS"
        if key in results:
            acc = results[key] * 100
            n   = sizes.get(lang_name, "?")
            print(f"{lang_name:12s} | {acc:8.1f}%    | {n}")

    torch.save(results, os.path.join(args.save_dir, "results.pt"))
    print(f"\nSaved to: {args.save_dir}/results.pt")


if __name__ == "__main__":
    main()