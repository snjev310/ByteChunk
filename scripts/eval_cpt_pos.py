# scripts/eval_cpt_pos.py
"""
Evaluate CPT baseline on POS tagging.
Loads CPT checkpoint, restores unfrozen layers,
runs standard subword probe (same as qwen_subword_pos_baseline.py).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_cpt_pos \
        --cpt_ckpt runs_cpt/Qwen_Qwen3-4B/best.pt \
        --train_data data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
        --test_langs \
            Bhojpuri:data/ud_bhojpuri/bho_bhtb-ud-test.conllu \
            Marathi:data/ud_marathi/mr_ufal-ud-train.conllu \
            Sanskrit:data/ud_sanskrit/sa_ufal-ud-test.conllu \
            Magahi:data/ud_magahi/mag_mgtb-ud-test.conllu \
            Urdu:data/ud_urdu/ur_udtb-ud-test.conllu \
        --save_dir runs_cpt/pos \
        --epochs 10
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

from configs.default import DEVICE, DTYPE, MODEL_ID


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


# ── Dataset ───────────────────────────────────────────────────────────────────
class SubwordPOSDataset(Dataset):
    def __init__(self, sentences, tag2id, tokenizer, max_len=512,
                 split="train", train_ratio=0.8, seed=42):
        self.tokenizer = tokenizer
        self.tag2id    = tag2id
        self.max_len   = max_len

        import random
        rng = random.Random(seed)
        data = sentences[:]
        rng.shuffle(data)
        cut = int(len(data) * train_ratio)
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
            if wid is None:
                continue
            if wid != prev_word_id and wid < len(tags):
                labels[i] = self.tag2id.get(tags[wid], 0)
            prev_word_id = wid

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── POS Head ─────────────────────────────────────────────────────────────────
class POSHead(nn.Module):
    def __init__(self, hidden_size, n_tags, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(hidden_size, n_tags)

    def forward(self, x):
        return self.proj(self.dropout(x))


# ── Load CPT model ────────────────────────────────────────────────────────────
def load_cpt_model(cpt_ckpt_path):
    ckpt   = torch.load(cpt_ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    model_id = config.get("model_id", MODEL_ID)

    print(f"  Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map="auto"
    )

    # Restore CPT weights
    print(f"  Restoring CPT weights from step {ckpt.get('global_step')}")
    model.load_state_dict(ckpt["model_state"], strict=False)

    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model, config


# ── Eval loop ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, pos_head, loader, hidden_size):
    model.eval(); pos_head.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        hidden  = outputs.hidden_states[-1]  # [B, T, D]
        logits  = pos_head(hidden)           # [B, T, n_tags]

        mask    = labels != -100
        preds   = logits.argmax(-1)
        correct += (preds[mask] == labels[mask]).sum().item()
        total   += mask.sum().item()

    return correct / total if total > 0 else 0.0


# ── Train probe ───────────────────────────────────────────────────────────────
def train_probe(model, pos_head, train_loader, val_loader, args):
    optimizer = torch.optim.AdamW(
        pos_head.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        pos_head.train()
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

            logits = pos_head(hidden)
            mask   = labels != -100
            loss   = F.cross_entropy(
                logits[mask], labels[mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pos_head.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        val_acc = evaluate(model, pos_head, val_loader,
                           hidden.shape[-1])
        print(f"  Epoch {epoch}/{args.epochs}  val_acc={val_acc*100:.1f}%")
        if val_acc > best_acc:
            best_acc = val_acc
    return best_acc


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_ckpt",   required=True)
    p.add_argument("--train_data", required=True)
    p.add_argument("--test_langs", nargs="+", required=True,
                   help="Name:path pairs e.g. Bhojpuri:data/...")
    p.add_argument("--save_dir",   default="runs_cpt/pos")
    p.add_argument("--max_len",    type=int,   default=256)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=2e-4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load CPT model
    print("Loading CPT model...")
    model, config = load_cpt_model(args.cpt_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model_id", MODEL_ID))
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "text_config", model.config).hidden_size

    # Load Hindi training data
    print("Loading Hindi training data...")
    hi_sentences = read_conll(args.train_data)
    all_tags = sorted({t for _, tags in hi_sentences for t in tags})
    tag2id   = {t: i for i, t in enumerate(all_tags)}
    n_tags   = len(tag2id)
    print(f"  Sentences: {len(hi_sentences)}  Tags: {n_tags}")

    train_ds = SubwordPOSDataset(hi_sentences, tag2id, tokenizer,
                                  args.max_len, "train")
    val_ds   = SubwordPOSDataset(hi_sentences, tag2id, tokenizer,
                                  args.max_len, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Train Hindi probe
    print("\nTraining Hindi probe...")
    pos_head = POSHead(hidden_size, n_tags).to(DEVICE, dtype=DTYPE)
    hindi_acc = train_probe(model, pos_head, train_loader,
                            val_loader, args)
    print(f"Hindi XL accuracy: {hindi_acc*100:.1f}%")

    results = {"Hindi_XL": hindi_acc}
    torch.save(tag2id, os.path.join(args.save_dir, "tag2id.pt"))

    # Evaluate on target languages
    for lang_spec in args.test_langs:
        lang, path = lang_spec.split(":", 1)
        print(f"\n{'='*50}")
        print(f"Evaluating: {lang}")

        lang_sentences = read_conll(path)
        print(f"  Sentences: {len(lang_sentences)}")

        # ZS: apply Hindi probe directly
        zs_ds = SubwordPOSDataset(lang_sentences, tag2id, tokenizer,
                                   args.max_len, "val",
                                   train_ratio=0.0)
        zs_loader = DataLoader(zs_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=0)
        zs_acc = evaluate(model, pos_head, zs_loader, hidden_size)
        print(f"  ZS accuracy: {zs_acc*100:.1f}%")

        # XL: fresh probe on 80% target data
        xl_train_ds = SubwordPOSDataset(lang_sentences, tag2id,
                                         tokenizer, args.max_len,
                                         "train")
        xl_val_ds   = SubwordPOSDataset(lang_sentences, tag2id,
                                         tokenizer, args.max_len,
                                         "val")
        xl_train_loader = DataLoader(xl_train_ds, batch_size=args.batch_size,
                                     shuffle=True,  num_workers=0)
        xl_val_loader   = DataLoader(xl_val_ds,   batch_size=args.batch_size,
                                     shuffle=False, num_workers=0)

        fresh_head = POSHead(hidden_size, n_tags).to(DEVICE, dtype=DTYPE)
        xl_acc = train_probe(model, fresh_head, xl_train_loader,
                             xl_val_loader, args)
        print(f"  XL accuracy: {xl_acc*100:.1f}%")

        results[f"{lang}_ZS"] = zs_acc
        results[f"{lang}_XL"] = xl_acc

    # Summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY — CPT Subword Baseline")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k}: {v*100:.1f}%")

    torch.save(results, os.path.join(args.save_dir, "results.pt"))
    print(f"\nSaved to: {args.save_dir}/results.pt")


if __name__ == "__main__":
    main()