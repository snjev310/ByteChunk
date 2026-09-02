# scripts/qwen_subword_ner_baseline.py
"""
Qwen subword NER baseline.

Protocol (matches eval_chunk_ner.py exactly):
  1. Train linear NER probe on frozen Qwen2.5-1.5B representations
     using Hindi WikiANN (train split, 80/20 split seed=42)
  2. Zero-shot (ZS): apply Hindi-trained probe directly to ALL
     target language sentences — no retraining
  3. Cross-lingual (XL): train fresh probe on 80% of target language,
     evaluate on 20%

Metric: entity-level F1 (seqeval)

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.qwen_subword_ner_baseline \
        --train_data data/wikiann/hindi/train.conll \
        --test_langs \
            Urdu:data/wikiann/urdu/test.conll \
            Marathi:data/wikiann/marathi/test.conll \
            Sanskrit:data/wikiann/sanskrit/test.conll \
        --save_dir   runs_qwen/qwen_subword_ner \
        --epochs     20
"""

import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report

from configs.default import MODEL_ID, DEVICE, DTYPE


# ── CoNLL NER reader ──────────────────────────────────────────────────────────

def read_ner_conll(path):
    sentences, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "":
                if words:
                    sentences.append((words[:], tags[:]))
                    words, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            words.append(parts[0])
            tags.append(parts[1])
    if words:
        sentences.append((words, tags))
    return sentences


# ── Dataset ───────────────────────────────────────────────────────────────────

class SubwordNERDataset(Dataset):
    """
    Tokenizes each sentence with Qwen tokenizer.
    Labels assigned to FIRST subword token of each word.
    Other subword tokens get IGNORE_IDX = -100.
    Stores original string tags for seqeval evaluation.
    """
    IGNORE_IDX = -100

    def __init__(self, sentences, tag2id, tokenizer, max_len=512,
                 split="train", train_ratio=0.8, seed=42):
        self.tag2id    = tag2id
        self.tokenizer = tokenizer
        self.max_len   = max_len

        rng  = random.Random(seed)
        data = sentences[:]
        rng.shuffle(data)
        cut  = int(len(data) * train_ratio)
        self.sentences = data[:cut] if split == "train" else data[cut:]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words, tags = self.sentences[idx]

        input_ids  = [self.tokenizer.bos_token_id or 1]
        label_ids  = [self.IGNORE_IDX]
        word_first = []   # position of first subword of each word

        for word, tag in zip(words, tags):
            toks = self.tokenizer.encode(
                " " + word, add_special_tokens=False)
            if not toks:
                continue
            tag_id = self.tag2id.get(tag, self.tag2id.get("O", 0))
            word_first.append(len(input_ids))
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
            "true_tags":      tags[:],
            "n_words":        len(words),
        }


def ner_collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "labels":         torch.stack([b["labels"]          for b in batch]),
        "true_tags":      [b["true_tags"] for b in batch],
        "n_words":        [b["n_words"]   for b in batch],
    }


# ── NER head ─────────────────────────────────────────────────────────────────

class SubwordNERHead(nn.Module):
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
            ignore_index=-100)


# ── F1 evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_f1(model, ner_head, loader, id2tag):
    model.eval(); ner_head.eval()
    all_true, all_pred = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        true_tags_list = batch["true_tags"]
        n_words_list   = batch["n_words"]

        encoder = getattr(model, "language_model", model)
        out    = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state.to(dtype=torch.float32)
        logits = ner_head(hidden)           # [B, T, n_tags]
        preds  = logits.argmax(-1)          # [B, T]

        for i in range(len(true_tags_list)):
            true_seq = true_tags_list[i]
            n_words  = n_words_list[i]

            # Extract predictions at first-subword positions
            # Labels tensor has -100 for non-first subwords
            lab_row  = labels[i]            # [T]
            pred_row = preds[i]             # [T]

            pred_seq = []
            for t in range(len(lab_row)):
                if lab_row[t].item() != -100:
                    pred_seq.append(
                        id2tag.get(pred_row[t].item(), "O"))
                if len(pred_seq) == n_words:
                    break

            # Pad or truncate to match true sequence length
            true_seq = true_seq[:n_words]
            if len(pred_seq) < len(true_seq):
                pred_seq += ["O"] * (len(true_seq) - len(pred_seq))
            pred_seq = pred_seq[:len(true_seq)]

            if true_seq and pred_seq:
                all_true.append(true_seq)
                all_pred.append(pred_seq)

    if not all_true:
        return 0.0, {}
    f1     = f1_score(all_true, all_pred, zero_division=0)
    report = classification_report(
        all_true, all_pred, output_dict=True, zero_division=0)
    return f1, report


# ── Train probe ───────────────────────────────────────────────────────────────

def train_probe(model, ner_head, train_loader, val_loader,
                id2tag, args, desc=""):
    optimizer = torch.optim.AdamW(
        ner_head.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        ner_head.train()
        total_loss, n_batches = 0.0, 0

        for batch in tqdm(train_loader,
                          desc=f"{desc} Epoch {epoch}/{args.epochs}",
                          leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            with torch.no_grad():
                out    = model(input_ids=input_ids,
                               attention_mask=attention_mask)
                hidden = out.last_hidden_state.to(dtype=torch.float32)

            logits = ner_head(hidden)
            loss   = ner_head.compute_loss(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ner_head.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); n_batches += 1

        val_f1, _ = evaluate_f1(model, ner_head, val_loader, id2tag)
        print(f"  {desc} Epoch {epoch:3d}/{args.epochs} | "
              f"loss={total_loss/max(n_batches,1):.4f} | "
              f"val_F1={val_f1*100:.1f}%")
        if val_f1 > best_f1:
            best_f1 = val_f1
    return best_f1


# ── Argparse ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data",  required=True,
                   help="Hindi WikiANN train.conll")
    p.add_argument("--test_langs",  nargs="+", default=[],
                   help="Name:path e.g. Urdu:data/wikiann/urdu/test.conll")
    p.add_argument("--save_dir",    default="runs_qwen/qwen_subword_ner")
    p.add_argument("--max_len",     type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=2e-4)
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

    # ── Load frozen Qwen ──────────────────────────────────────────────────
    print(f"Loading Qwen: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    hidden_size = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    print(f"  Hidden size: {hidden_size}  |  All params frozen")

    # ── Build tag vocab from Hindi ─────────────────────────────────────────
    print(f"\nLoading Hindi NER data: {args.train_data}")
    hindi_sents = read_ner_conll(args.train_data)
    all_tags    = sorted({t for _, tags in hindi_sents for t in tags})
    tag2id      = {t: i for i, t in enumerate(all_tags)}
    id2tag      = {i: t for t, i in tag2id.items()}
    n_tags      = len(tag2id)
    print(f"  Sentences: {len(hindi_sents)}  |  Tags ({n_tags}): {all_tags}")
    torch.save(tag2id, os.path.join(args.save_dir, "tag2id.pt"))

    # ── Train Hindi probe ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1: Train NER probe on Hindi")
    print(f"{'='*60}")

    train_ds = SubwordNERDataset(
        hindi_sents, tag2id, tokenizer, args.max_len,
        "train", 0.8, 42)
    val_ds   = SubwordNERDataset(
        hindi_sents, tag2id, tokenizer, args.max_len,
        "val",   0.8, 42)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=ner_collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=ner_collate_fn,
                              num_workers=0)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    ner_head = SubwordNERHead(hidden_size, n_tags).to(DEVICE)
    print(f"  NER head params: "
          f"{sum(p.numel() for p in ner_head.parameters()):,}")

    best_hindi_f1 = train_probe(
        model, ner_head, train_loader, val_loader,
        id2tag, args, "Hindi")
    print(f"\n  Hindi best val F1: {best_hindi_f1*100:.1f}%")

    results = {"Hindi_XL_F1": best_hindi_f1}

    # ── Cross-lingual evaluation ───────────────────────────────────────────
    def eval_language(lang_name, lang_path, hindi_probe):
        print(f"\n{'='*60}")
        print(f"{lang_name.upper()} EVALUATION")
        print(f"{'='*60}")

        lang_sents = read_ner_conll(lang_path)
        print(f"  Total sentences: {len(lang_sents)}")

        # ── ZS: Hindi probe → all target data ─────────────────────────────
        print(f"\n  [ZS] Zero-shot: Hindi probe on all {lang_name} data")
        zs_ds = SubwordNERDataset(
            lang_sents, tag2id, tokenizer, args.max_len,
            split="val", train_ratio=0.0, seed=42)
        zs_loader = DataLoader(zs_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=ner_collate_fn,
                               num_workers=0)
        zs_f1, _ = evaluate_f1(model, hindi_probe, zs_loader, id2tag)
        print(f"  [ZS] {lang_name} F1: {zs_f1*100:.1f}%")

        # ── XL: fresh probe on 80% target data ────────────────────────────
        print(f"\n  [XL] Cross-lingual: fresh probe on 80% {lang_name}")
        xl_train_ds = SubwordNERDataset(
            lang_sents, tag2id, tokenizer, args.max_len,
            "train", 0.8, 42)
        xl_val_ds   = SubwordNERDataset(
            lang_sents, tag2id, tokenizer, args.max_len,
            "val",   0.8, 42)
        xl_train_loader = DataLoader(
            xl_train_ds, batch_size=args.batch_size,
            shuffle=True,  collate_fn=ner_collate_fn, num_workers=0)
        xl_val_loader   = DataLoader(
            xl_val_ds,   batch_size=args.batch_size,
            shuffle=False, collate_fn=ner_collate_fn, num_workers=0)
        print(f"  Train: {len(xl_train_ds)} | Val: {len(xl_val_ds)}")

        fresh_head = SubwordNERHead(hidden_size, n_tags).to(DEVICE)
        xl_f1 = train_probe(
            model, fresh_head, xl_train_loader, xl_val_loader,
            id2tag, args, lang_name)
        print(f"  [XL] {lang_name} best F1: {xl_f1*100:.1f}%")

        return zs_f1, xl_f1

    for lang_name, lang_path in test_langs.items():
        zs_f1, xl_f1 = eval_language(lang_name, lang_path, ner_head)
        results[f"{lang_name}_ZS_F1"] = zs_f1
        results[f"{lang_name}_XL_F1"] = xl_f1

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — Qwen Subword NER (entity F1)")
    print(f"{'='*60}")
    print(f"  Encoder: Qwen2.5-1.5B fully frozen")
    print(f"  ZS = zero-shot  |  XL = cross-lingual (80% target data)")
    print()
    print(f"{'Language':12s} | {'ZS F1':10s} | {'XL F1':10s}")
    print("-"*38)
    print(f"{'Hindi':12s} | {'---':10s} | {best_hindi_f1*100:6.1f}%")
    for lang_name in test_langs:
        zs = results.get(f"{lang_name}_ZS_F1", 0) * 100
        xl = results.get(f"{lang_name}_XL_F1", 0) * 100
        print(f"{lang_name:12s} | {zs:6.1f}%    | {xl:6.1f}%")

    torch.save(results, os.path.join(args.save_dir, "results.pt"))
    print(f"\nSaved to: {args.save_dir}/results.pt")


if __name__ == "__main__":
    main()