# scripts/eval_chunk_ner.py
"""
Chunk-level NER evaluation for H-Net.

Protocol:
  ZS: Train NER probe on Hindi WikiANN, apply Hindi probe directly
      to ALL target language sentences (no retraining).
  XL: Train fresh probe on 80% of target language data, eval on 20%.

Labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC (WikiANN 7-class)
Metric: entity-level F1 (seqeval)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_chunk_ner \
        --hnet_ckpt  runs_qwen/hnet_pretrain_pos_guided/step_7000.pt \
        --train_data data/wikiann/hindi/train.conll \
        --test_langs \
            Urdu:data/wikiann/urdu/test.conll \
            Marathi:data/wikiann/marathi/test.conll \
            Sanskrit:data/wikiann/sanskrit/test.conll \
        --save_dir   runs_qwen/hnet_chunk_ner \
        --epochs     20 --n_local_layers 1
"""

import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report

from models.load_hnet import load_hnet_encoder
from configs.default  import DEVICE, DTYPE, MODEL_ID


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

class ChunkNERDataset(Dataset):
    """
    Encodes each sentence as UTF-8 bytes.
    Stores word-level NER tags and word boundary positions.
    One label per word — aligned to chunks during forward pass.
    """
    IGNORE_IDX = -100
    PAD_ID     = 0
    EOS_ID     = 1

    def __init__(self, sentences, tag2id, max_len=512,
                 split="train", train_ratio=0.8, seed=42):
        self.tag2id  = tag2id
        self.max_len = max_len

        rng  = random.Random(seed)
        data = sentences[:]
        rng.shuffle(data)
        cut  = int(len(data) * train_ratio)
        self.sentences = data[:cut] if split == "train" else data[cut:]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words, tags = self.sentences[idx]
        byte_ids, word_ids, word_tags = [], [], []

        for w_idx, (word, tag) in enumerate(zip(words, tags)):
            tag_id = self.tag2id.get(tag, self.tag2id.get("O", 0))
            word_tags.append(tag_id)
            for b in word.encode("utf-8"):
                byte_ids.append(b)
                word_ids.append(w_idx)
            # space between words
            byte_ids.append(ord(" "))
            word_ids.append(-1)

        # Truncate
        byte_ids = byte_ids[:self.max_len - 1]
        word_ids = word_ids[:self.max_len - 1]
        byte_ids.append(self.EOS_ID); word_ids.append(-1)

        # Pad
        pad_len   = self.max_len - len(byte_ids)
        byte_ids += [self.PAD_ID] * pad_len
        word_ids += [-1]          * pad_len

        return {
            "input_ids":      torch.tensor(byte_ids,  dtype=torch.long),
            "attention_mask": (torch.tensor(byte_ids,
                               dtype=torch.long) != self.PAD_ID).long(),
            "word_ids":       torch.tensor(word_ids,  dtype=torch.long),
            "word_tags":      torch.tensor(word_tags, dtype=torch.long),
            "n_words":        len(words),
            "true_tags":      tags[:],   # original string tags for seqeval
        }


def ner_collate_fn(batch):
    max_words   = max(b["n_words"] for b in batch)
    padded_tags = torch.full((len(batch), max_words), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        padded_tags[i, :b["n_words"]] = b["word_tags"]
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "word_ids":       torch.stack([b["word_ids"]        for b in batch]),
        "word_tags":      padded_tags,
        "n_words":        [b["n_words"]  for b in batch],
        "true_tags":      [b["true_tags"] for b in batch],
    }


# ── Chunk-word alignment ──────────────────────────────────────────────────────

def get_chunk_word_labels(chunk_ids_batch, word_ids_batch,
                          word_tags_batch, n_words_list):
    """Returns List[Tensor[C_i]] — NER label per chunk (-100 if no word)."""
    chunk_labels_batch = []
    for i in range(len(chunk_ids_batch)):
        chunk_ids = chunk_ids_batch[i]
        word_ids  = word_ids_batch[i]
        word_tags = word_tags_batch[i]
        C_i = chunk_ids.max().item() + 1 if len(chunk_ids) > 0 else 0

        chunk_word_votes = {}
        for t in range(min(len(chunk_ids), word_ids.shape[0])):
            c = chunk_ids[t].item()
            w = word_ids[t].item()
            if w >= 0 and c >= 0:
                chunk_word_votes.setdefault(c, {})
                chunk_word_votes[c][w] = \
                    chunk_word_votes[c].get(w, 0) + 1

        labels = torch.full((C_i,), -100, dtype=torch.long)
        for c, votes in chunk_word_votes.items():
            if c < C_i and votes:
                best_w = max(votes, key=votes.get)
                if best_w < word_tags.shape[0]:
                    labels[c] = word_tags[best_w]
        chunk_labels_batch.append(labels)
    return chunk_labels_batch


def chunk_to_word_preds(chunk_ids, word_ids, chunk_preds, n_words, id2tag):
    """
    Map chunk-level predictions back to word-level string tags.

    For each word, find the chunk that covers it (first byte match)
    and use that chunk's predicted tag.
    Words not covered by any valid chunk get tag "O".

    Returns: List[str] of length n_words
    """
    # Build word → chunk mapping (first chunk that covers each word)
    word_to_chunk = {}
    for t in range(min(len(chunk_ids), len(word_ids))):
        c = chunk_ids[t].item()
        w = word_ids[t].item()
        if w >= 0 and c >= 0 and w not in word_to_chunk:
            word_to_chunk[w] = c

    pred_tags = []
    for w in range(n_words):
        if w in word_to_chunk:
            c = word_to_chunk[w]
            if c < len(chunk_preds):
                pred_tags.append(id2tag.get(chunk_preds[c].item(), "O"))
            else:
                pred_tags.append("O")
        else:
            pred_tags.append("O")
    return pred_tags


# ── NER head ─────────────────────────────────────────────────────────────────

class ChunkNERHead(nn.Module):
    def __init__(self, d_model, n_tags, dropout=0.1):
        super().__init__()
        hidden          = min(256, d_model // 4)
        self.dropout    = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(d_model, hidden)
        self.act        = nn.GELU()
        self.norm       = nn.LayerNorm(hidden)
        self.proj       = nn.Linear(hidden, n_tags)

    def forward(self, x):
        h = self.dropout(x)
        h = self.act(self.bottleneck(h))
        h = self.norm(h)
        return self.proj(h)

    def compute_loss(self, chunk_embs_batch, chunk_labels_batch):
        total_loss, total_count = \
            torch.tensor(0.0, device=chunk_embs_batch[0].device), 0
        for embs, labels in zip(chunk_embs_batch, chunk_labels_batch):
            labels = labels.to(embs.device)
            mask   = labels != -100
            if mask.sum() == 0: continue
            logits = self.forward(embs)
            loss   = F.cross_entropy(
                logits[mask], labels[mask], reduction="sum")
            total_loss  = total_loss + loss
            total_count += mask.sum().item()
        return total_loss / max(total_count, 1)


# ── Encoder forward ───────────────────────────────────────────────────────────

@torch.no_grad()
def get_chunk_embs(encoder, input_ids, attention_mask):
    h_hat, p, aux = encoder(input_ids, attention_mask)
    chunk_out      = aux["chunk_out"]
    chunk_embs     = chunk_out.get("chunk_embs_smooth",
                                   chunk_out["chunk_embs"])
    chunk_ids      = chunk_out["chunk_ids"]
    return chunk_embs, chunk_ids


# ── F1 evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_f1(encoder, ner_head, loader, id2tag):
    """
    Evaluate NER using seqeval entity-level F1.
    Correctly aligns chunk predictions back to word-level string tags.
    """
    encoder.eval(); ner_head.eval()
    all_true, all_pred = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        word_ids       = batch["word_ids"]      # keep on CPU for indexing
        word_tags      = batch["word_tags"].to(DEVICE)
        n_words_list   = batch["n_words"]
        true_tags_list = batch["true_tags"]

        chunk_embs_batch, chunk_ids_batch = get_chunk_embs(
            encoder, input_ids, attention_mask)
        chunk_embs_batch = [c.to(DEVICE, dtype=DTYPE)
                            for c in chunk_embs_batch]

        for i, (embs, chunk_ids) in enumerate(
                zip(chunk_embs_batch, chunk_ids_batch)):

            # Get chunk-level predictions
            logits      = ner_head(embs)          # [C, n_tags]
            chunk_preds = logits.argmax(-1)        # [C]

            # Map chunk predictions → word-level string tags
            n_words  = n_words_list[i]
            wid_row  = word_ids[i]                 # [T] on CPU
            pred_seq = chunk_to_word_preds(
                chunk_ids, wid_row, chunk_preds, n_words, id2tag)

            # Ground truth string tags
            true_seq = true_tags_list[i][:n_words]

            # Sanity check lengths
            assert len(pred_seq) == len(true_seq), \
                f"Length mismatch: pred={len(pred_seq)} true={len(true_seq)}"

            all_true.append(true_seq)
            all_pred.append(pred_seq)

    if not all_true:
        return 0.0, {}
    f1     = f1_score(all_true, all_pred)
    report = classification_report(all_true, all_pred,
                                   output_dict=True, zero_division=0)
    return f1, report


# ── Train probe ───────────────────────────────────────────────────────────────

def train_probe(encoder, ner_head, train_loader, val_loader,
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
            word_ids       = batch["word_ids"].to(DEVICE)
            word_tags      = batch["word_tags"].to(DEVICE)

            chunk_embs_batch, chunk_ids_batch = get_chunk_embs(
                encoder, input_ids, attention_mask)
            chunk_labels_batch = get_chunk_word_labels(
                chunk_ids_batch, word_ids, word_tags, batch["n_words"])
            chunk_embs_batch = [c.to(DEVICE, dtype=DTYPE)
                                for c in chunk_embs_batch]

            loss = ner_head.compute_loss(
                chunk_embs_batch, chunk_labels_batch)
            if torch.isnan(loss):
                optimizer.zero_grad(); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ner_head.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); n_batches += 1

        val_f1, _ = evaluate_f1(
            encoder, ner_head, val_loader, id2tag)
        print(f"  {desc} Epoch {epoch:3d}/{args.epochs} | "
              f"loss={total_loss/max(n_batches,1):.4f} | "
              f"val_F1={val_f1*100:.1f}%")
        if val_f1 > best_f1:
            best_f1 = val_f1
    return best_f1


# ── Argparse ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hnet_ckpt",      required=True)
    p.add_argument("--train_data",     required=True,
                   help="Hindi WikiANN train.conll")
    p.add_argument("--test_langs",     nargs="+", default=[],
                   help="Name:path e.g. Urdu:data/wikiann/urdu/test.conll")
    p.add_argument("--save_dir",       default="runs_qwen/hnet_chunk_ner")
    p.add_argument("--max_len",        type=int,   default=512)
    p.add_argument("--batch_size",     type=int,   default=4)
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

    train_ds = ChunkNERDataset(hindi_sents, tag2id, args.max_len,
                               "train", 0.8, 42)
    val_ds   = ChunkNERDataset(hindi_sents, tag2id, args.max_len,
                               "val",   0.8, 42)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=ner_collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=ner_collate_fn,
                              num_workers=0)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    ner_head = ChunkNERHead(d_model, n_tags).to(DEVICE, dtype=DTYPE)
    print(f"  NER head params: "
          f"{sum(p.numel() for p in ner_head.parameters()):,}")

    best_hindi_f1 = train_probe(
        encoder, ner_head, train_loader, val_loader,
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
        zs_ds = ChunkNERDataset(
            lang_sents, tag2id, args.max_len,
            split="val", train_ratio=0.0, seed=42)
        zs_loader = DataLoader(zs_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=ner_collate_fn,
                               num_workers=0)
        zs_f1, zs_report = evaluate_f1(
            encoder, hindi_probe, zs_loader, id2tag)
        print(f"  [ZS] {lang_name} F1: {zs_f1*100:.1f}%")

        # ── XL: fresh probe on 80% target data ────────────────────────────
        print(f"\n  [XL] Cross-lingual: fresh probe on 80% {lang_name}")
        xl_train_ds = ChunkNERDataset(
            lang_sents, tag2id, args.max_len, "train", 0.8, 42)
        xl_val_ds   = ChunkNERDataset(
            lang_sents, tag2id, args.max_len, "val",   0.8, 42)
        xl_train_loader = DataLoader(
            xl_train_ds, batch_size=args.batch_size,
            shuffle=True,  collate_fn=ner_collate_fn, num_workers=0)
        xl_val_loader   = DataLoader(
            xl_val_ds,   batch_size=args.batch_size,
            shuffle=False, collate_fn=ner_collate_fn, num_workers=0)
        print(f"  Train: {len(xl_train_ds)} | Val: {len(xl_val_ds)}")

        fresh_head = ChunkNERHead(d_model, n_tags).to(DEVICE, dtype=DTYPE)
        xl_f1 = train_probe(
            encoder, fresh_head, xl_train_loader, xl_val_loader,
            id2tag, args, lang_name)
        print(f"  [XL] {lang_name} best F1: {xl_f1*100:.1f}%")

        return zs_f1, xl_f1

    for lang_name, lang_path in test_langs.items():
        zs_f1, xl_f1 = eval_language(lang_name, lang_path, ner_head)
        results[f"{lang_name}_ZS_F1"] = zs_f1
        results[f"{lang_name}_XL_F1"] = xl_f1

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — H-Net Chunk-Level NER (entity F1)")
    print(f"{'='*60}")
    print(f"  Encoder: frozen, pretrained on Hindi only")
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