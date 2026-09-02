# scripts/eval_cpt_ner.py
"""
Evaluate CPT baseline on NER using WikiANN.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.eval_cpt_ner \
        --cpt_ckpt   runs_cpt/Qwen_Qwen3-4B/best.pt \
        --train_data data/wikiann/hindi/train.conll \
        --test_langs \
            Urdu:data/wikiann/urdu/test.conll \
            Marathi:data/wikiann/marathi/test.conll \
            Sanskrit:data/wikiann/sanskrit/test.conll \
        --save_dir   runs_cpt/ner \
        --epochs     20
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from seqeval.metrics import f1_score

from configs.default import DEVICE, DTYPE, MODEL_ID


# ── CoNLL Reader ──────────────────────────────────────────────────────────────
def read_conll_ner(path):
    sentences, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "" or line.startswith("#"):
                if words:
                    sentences.append((words[:], tags[:]))
                    words, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2: continue
            words.append(parts[0])
            tags.append(parts[-1])
    if words:
        sentences.append((words, tags))
    return sentences


# ── Dataset ───────────────────────────────────────────────────────────────────
class SubwordNERDataset(Dataset):
    def __init__(self, sentences, tag2id, tokenizer, max_len=256,
                 split="train", train_ratio=0.8, seed=42):
        self.tokenizer = tokenizer
        self.tag2id    = tag2id
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
        encoding = self.tokenizer(
            words,
            is_split_into_words = True,
            max_length          = self.max_len,
            truncation          = True,
            padding             = "max_length",
            return_tensors      = "pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids       = encoding.word_ids()

        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        prev_word_id = None
        for i, wid in enumerate(word_ids):
            if wid is None: continue
            if wid != prev_word_id and wid < len(tags):
                labels[i] = self.tag2id.get(tags[wid], 0)
            prev_word_id = wid

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "true_tags":      tags[:],
            "word_ids":       [w if w is not None else -1
                               for w in word_ids],
        }


def ner_collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "labels":         torch.stack([b["labels"]          for b in batch]),
        "true_tags":      [b["true_tags"] for b in batch],
        "word_ids":       [b["word_ids"]  for b in batch],
    }


# ── NER Head ──────────────────────────────────────────────────────────────────
class NERHead(nn.Module):
    def __init__(self, hidden_size, n_tags, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(hidden_size, n_tags)

    def forward(self, x):
        return self.proj(self.dropout(x))


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


# ── Evaluate F1 ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_f1(model, ner_head, loader, id2tag):
    model.eval(); ner_head.eval()
    all_true, all_pred = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        true_tags      = batch["true_tags"]
        word_ids_batch = batch["word_ids"]

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)
        hidden  = outputs.hidden_states[-1]
        logits  = ner_head(hidden).argmax(-1).cpu()

        for i in range(len(true_tags)):
            word_ids  = word_ids_batch[i]
            n_words   = len(true_tags[i])
            pred_tags = []
            seen      = set()
            for j, wid in enumerate(word_ids):
                if wid == -1 or wid in seen or wid >= n_words:
                    continue
                pred_tags.append(id2tag[logits[i][j].item()])
                seen.add(wid)
            # Pad/trim to match true length
            while len(pred_tags) < n_words:
                pred_tags.append("O")
            pred_tags = pred_tags[:n_words]
            all_true.append(true_tags[i])
            all_pred.append(pred_tags)

    return f1_score(all_true, all_pred) * 100


# ── Train probe ───────────────────────────────────────────────────────────────
def train_probe(model, ner_head, train_loader, val_loader,
                id2tag, args):
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
        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch}/{args.epochs}",
                          leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
                hidden = outputs.hidden_states[-1]

            logits = ner_head(hidden)
            mask   = labels != -100
            loss   = F.cross_entropy(logits[mask], labels[mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ner_head.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        val_f1 = evaluate_f1(model, ner_head, val_loader, id2tag)
        print(f"  Epoch {epoch}/{args.epochs}  val_F1={val_f1:.1f}%")
        if val_f1 > best_f1:
            best_f1 = val_f1
    return best_f1


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_ckpt",   required=True)
    p.add_argument("--train_data", required=True)
    p.add_argument("--test_langs", nargs="+", required=True)
    p.add_argument("--save_dir",   default="runs_cpt/ner")
    p.add_argument("--max_len",    type=int,   default=256)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=2e-4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading CPT model...")
    model, config = load_cpt_model(args.cpt_ckpt)
    tokenizer  = AutoTokenizer.from_pretrained(
        config.get("model_id", MODEL_ID))
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "text_config", model.config).hidden_size

    print("Loading Hindi NER data...")
    hi_sentences = read_conll_ner(args.train_data)
    all_tags     = sorted({t for _, tags in hi_sentences for t in tags})
    tag2id       = {t: i for i, t in enumerate(all_tags)}
    id2tag       = {i: t for t, i in tag2id.items()}
    n_tags       = len(tag2id)
    print(f"  Sentences: {len(hi_sentences)}  Tags: {n_tags}")

    train_ds = SubwordNERDataset(hi_sentences, tag2id, tokenizer,
                                  args.max_len, "train")
    val_ds   = SubwordNERDataset(hi_sentences, tag2id, tokenizer,
                                  args.max_len, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=ner_collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=ner_collate_fn,
                              num_workers=0)

    print("\nTraining Hindi NER probe...")
    ner_head   = NERHead(hidden_size, n_tags).to(DEVICE, dtype=DTYPE)
    hindi_f1   = train_probe(model, ner_head, train_loader,
                              val_loader, id2tag, args)
    print(f"Hindi XL F1: {hindi_f1:.1f}%")

    results = {"Hindi_XL": hindi_f1}
    torch.save(tag2id, os.path.join(args.save_dir, "tag2id.pt"))

    for lang_spec in args.test_langs:
        lang, path = lang_spec.split(":", 1)
        print(f"\n{'='*50}")
        print(f"Evaluating: {lang}")

        lang_sentences = read_conll_ner(path)
        print(f"  Sentences: {len(lang_sentences)}")

        # ZS
        zs_ds = SubwordNERDataset(lang_sentences, tag2id, tokenizer,
                                   args.max_len, "val",
                                   train_ratio=0.0)
        zs_loader = DataLoader(zs_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=ner_collate_fn,
                               num_workers=0)
        zs_f1 = evaluate_f1(model, ner_head, zs_loader, id2tag)
        print(f"  ZS F1: {zs_f1:.1f}%")

        # XL
        xl_train_ds = SubwordNERDataset(lang_sentences, tag2id,
                                         tokenizer, args.max_len, "train")
        xl_val_ds   = SubwordNERDataset(lang_sentences, tag2id,
                                         tokenizer, args.max_len, "val")
        xl_train_loader = DataLoader(xl_train_ds, batch_size=args.batch_size,
                                     shuffle=True,  collate_fn=ner_collate_fn,
                                     num_workers=0)
        xl_val_loader   = DataLoader(xl_val_ds,   batch_size=args.batch_size,
                                     shuffle=False, collate_fn=ner_collate_fn,
                                     num_workers=0)
        fresh_head = NERHead(hidden_size, n_tags).to(DEVICE, dtype=DTYPE)
        xl_f1 = train_probe(model, fresh_head, xl_train_loader,
                             xl_val_loader, id2tag, args)
        print(f"  XL F1: {xl_f1:.1f}%")

        results[f"{lang}_ZS"] = zs_f1
        results[f"{lang}_XL"] = xl_f1

    print(f"\n{'='*50}")
    print("FINAL SUMMARY — CPT NER Baseline")
    for k, v in results.items():
        print(f"  {k}: {v:.1f}%")

    torch.save(results, os.path.join(args.save_dir, "results.pt"))
    print(f"\nSaved to: {args.save_dir}/results.pt")


if __name__ == "__main__":
    main()