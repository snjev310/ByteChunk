# # scripts/eval_chunk_pos.py
# """
# Chunk-level POS tagging evaluation.

# Instead of operating on h_hat [B, T, D] (byte-level representations),
# this script operates directly on chunk embeddings [C_i, D] — the word-level
# representations produced by H-Net before dechunking.

# Motivation:
#   - H-Net chunks ≈ Hindi words (boundary rate 0.065)
#   - POS tagging is a word-level decision
#   - Dechunking averages chunk info back to bytes — may dilute signal
#   - Chunk embeddings are the most linguistically meaningful representations

# Pipeline:
#   bytes → local encoder → routing → chunking → smoother → chunk_embs [C_i, D]
#                                                                ↓
#                                                     POS head (linear probe)
#                                                                ↓
#                                                     tag per chunk (≈ per word)

# Alignment:
#   For each chunk, find which CoNLL word it corresponds to using byte positions.
#   Assign that word's POS tag as the chunk's label.

# Usage:
#     CUDA_VISIBLE_DEVICES=2 python -m scripts.eval_chunk_pos \
#         --hnet_ckpt runs_qwen/hnet_pretrain_pos_guided/step_7000.pt \
#         --data      data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#         --save_dir  runs_qwen/hnet_chunk_pos \
#         --epochs    20
# """

# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from transformers import get_cosine_schedule_with_warmup

# from models.load_hnet import load_hnet_encoder
# from configs.default  import DEVICE, DTYPE, MODEL_ID


# # ── CoNLL-U Reader ────────────────────────────────────────────────────────────

# def read_conll(path):
#     sentences, words, tags = [], [], []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.rstrip()
#             if line == "" or line.startswith("#"):
#                 if words:
#                     sentences.append((words[:], tags[:]))
#                     words, tags = [], []
#                 continue
#             parts = line.split("\t")
#             if len(parts) < 4: continue
#             if "-" in parts[0] or "." in parts[0]: continue
#             if parts[3] == "_": continue
#             words.append(parts[1])
#             tags.append(parts[3])
#     if words:
#         sentences.append((words, tags))
#     return sentences


# # ── Chunk-level POS Dataset ───────────────────────────────────────────────────

# class ChunkPOSDataset(Dataset):
#     """
#     Dataset for chunk-level POS tagging.

#     Each item is a sentence encoded as bytes.
#     Labels are at WORD level — one label per word (not per byte).
#     The dataset stores raw byte sequences and word boundaries.
#     The POS head assigns one tag per chunk during training.

#     Alignment during forward pass:
#       chunk_ids[t] tells which chunk byte t belongs to
#       word_ids[t]  tells which word byte t belongs to
#       → chunk_to_word[c] = most common word in chunk c
#       → chunk_label[c]   = tag of that word
#     """
#     IGNORE_IDX = -100
#     PAD_ID     = 0
#     EOS_ID     = 1

#     def __init__(self, sentences, tag2id, max_len=512,
#                  split="train", train_ratio=0.8, seed=42):
#         self.tag2id  = tag2id
#         self.max_len = max_len

#         import random
#         rng = random.Random(seed)
#         rng.shuffle(sentences)
#         cut = int(len(sentences) * train_ratio)
#         self.sentences = sentences[:cut] if split == "train" else sentences[cut:]

#     def __len__(self):
#         return len(self.sentences)

#     def __getitem__(self, idx):
#         words, tags = self.sentences[idx]

#         byte_ids  = []
#         word_ids  = []   # which word each byte belongs to (-1 = space/pad)
#         word_tags = []   # tag per word

#         for w_idx, (word, tag) in enumerate(zip(words, tags)):
#             word_bytes = list(word.encode("utf-8"))
#             tag_id     = self.tag2id.get(tag, 0)
#             word_tags.append(tag_id)

#             for b in word_bytes:
#                 byte_ids.append(b)
#                 word_ids.append(w_idx)

#             # Space between words
#             byte_ids.append(ord(" "))
#             word_ids.append(-1)   # space → no word

#         # Truncate
#         byte_ids  = byte_ids[:self.max_len - 1]
#         word_ids  = word_ids[:self.max_len - 1]

#         # EOS
#         byte_ids.append(self.EOS_ID)
#         word_ids.append(-1)

#         # Pad
#         pad_len   = self.max_len - len(byte_ids)
#         byte_ids += [self.PAD_ID] * pad_len
#         word_ids += [-1]          * pad_len

#         return {
#             "input_ids":      torch.tensor(byte_ids,  dtype=torch.long),
#             "attention_mask": (torch.tensor(byte_ids, dtype=torch.long) != self.PAD_ID).long(),
#             "word_ids":       torch.tensor(word_ids,  dtype=torch.long),
#             "word_tags":      torch.tensor(word_tags, dtype=torch.long),  # [n_words]
#             "n_words":        len(words),
#         }


# def chunk_collate_fn(batch):
#     """Custom collate — word_tags have variable length so pad them."""
#     max_words = max(b["n_words"] for b in batch)
#     padded_tags = torch.full((len(batch), max_words), -100, dtype=torch.long)
#     for i, b in enumerate(batch):
#         n = b["n_words"]
#         padded_tags[i, :n] = b["word_tags"]

#     return {
#         "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
#         "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
#         "word_ids":       torch.stack([b["word_ids"]        for b in batch]),
#         "word_tags":      padded_tags,
#         "n_words":        [b["n_words"] for b in batch],
#     }


# # ── Chunk-level alignment ─────────────────────────────────────────────────────

# def get_chunk_word_labels(chunk_ids_batch, word_ids_batch, word_tags_batch, n_words_list):
#     """
#     For each chunk, find which word it belongs to and return that word's POS tag.

#     chunk_ids_batch: List[Tensor[T_i]] — which chunk each byte belongs to
#     word_ids_batch:  Tensor[B, T]      — which word each byte belongs to (-1=space)
#     word_tags_batch: Tensor[B, max_W]  — POS tag per word
#     n_words_list:    List[int]         — number of words per sentence

#     Returns: List[Tensor[C_i]] — POS label per chunk (-100 if ambiguous)
#     """
#     B = len(chunk_ids_batch)
#     chunk_labels_batch = []

#     for i in range(B):
#         chunk_ids = chunk_ids_batch[i]   # [T_i]
#         word_ids  = word_ids_batch[i]    # [T] — full padded
#         word_tags = word_tags_batch[i]   # [max_W]
#         T_i       = len(chunk_ids)
#         C_i       = chunk_ids.max().item() + 1 if len(chunk_ids) > 0 else 0

#         # For each chunk, count how many bytes from each word it contains
#         chunk_word_votes = {}
#         for t in range(min(T_i, word_ids.shape[0])):
#             c = chunk_ids[t].item()
#             w = word_ids[t].item()
#             if w >= 0 and c >= 0:   # valid byte (not space/pad)
#                 if c not in chunk_word_votes:
#                     chunk_word_votes[c] = {}
#                 chunk_word_votes[c][w] = chunk_word_votes[c].get(w, 0) + 1

#         # Assign each chunk the tag of its majority word
#         labels = torch.full((C_i,), -100, dtype=torch.long)
#         for c, votes in chunk_word_votes.items():
#             if c < C_i and votes:
#                 best_word = max(votes, key=votes.get)
#                 if best_word < word_tags.shape[0]:
#                     labels[c] = word_tags[best_word]

#         chunk_labels_batch.append(labels)

#     return chunk_labels_batch


# # ── POS Head (chunk-level) ────────────────────────────────────────────────────

# class ChunkPOSHead(nn.Module):
#     """
#     Linear probe on chunk embeddings.
#     Same bottleneck architecture as byte-level POSHead for fair comparison.
#     d_model → 256 → n_tags
#     """
#     def __init__(self, d_model, n_tags, dropout=0.1):
#         super().__init__()
#         hidden          = min(256, d_model // 4)
#         self.dropout    = nn.Dropout(dropout)
#         self.bottleneck = nn.Linear(d_model, hidden)
#         self.act        = nn.GELU()
#         self.norm       = nn.LayerNorm(hidden)
#         self.proj       = nn.Linear(hidden, n_tags)

#     def forward(self, chunk_embs):
#         """chunk_embs: [C, D]  →  logits: [C, n_tags]"""
#         h = self.dropout(chunk_embs)
#         h = self.act(self.bottleneck(h))
#         h = self.norm(h)
#         return self.proj(h)

#     def compute_loss(self, chunk_embs_batch, chunk_labels_batch):
#         """
#         chunk_embs_batch:  List[Tensor[C_i, D]]
#         chunk_labels_batch: List[Tensor[C_i]]
#         """
#         total_loss  = torch.tensor(0.0, device=chunk_embs_batch[0].device)
#         total_count = 0

#         for embs, labels in zip(chunk_embs_batch, chunk_labels_batch):
#             labels = labels.to(embs.device)
#             mask   = labels != -100
#             if mask.sum() == 0:
#                 continue
#             logits = self.forward(embs)          # [C_i, n_tags]
#             loss   = F.cross_entropy(
#                 logits[mask], labels[mask], reduction="sum"
#             )
#             total_loss  = total_loss + loss
#             total_count += mask.sum().item()

#         return total_loss / max(total_count, 1)

#     @torch.no_grad()
#     def accuracy(self, chunk_embs_batch, chunk_labels_batch):
#         correct, total = 0, 0
#         for embs, labels in zip(chunk_embs_batch, chunk_labels_batch):
#             labels = labels.to(embs.device)
#             mask   = labels != -100
#             if mask.sum() == 0:
#                 continue
#             preds   = self.forward(embs).argmax(-1)
#             correct += (preds[mask] == labels[mask]).sum().item()
#             total   += mask.sum().item()
#         return correct / total if total > 0 else 0.0


# # ── Get chunk embeddings from encoder ────────────────────────────────────────

# @torch.no_grad()
# def get_chunk_embs(encoder, input_ids, attention_mask):
#     """
#     Run encoder and return smoothed chunk embeddings + chunk_ids.
#     These are the word-level representations BEFORE dechunking.
#     """
#     h_hat, p, aux = encoder(input_ids, attention_mask)
#     chunk_out      = aux["chunk_out"]
#     chunk_embs     = chunk_out.get("chunk_embs_smooth", chunk_out["chunk_embs"])
#     chunk_ids      = chunk_out["chunk_ids"]
#     return chunk_embs, chunk_ids


# # ── Argparse ─────────────────────────────────────────────────────────────────

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--hnet_ckpt",      required=True)
#     p.add_argument("--data",           required=True)
#     p.add_argument("--save_dir",       default="runs/hnet_chunk_pos")
#     p.add_argument("--max_len",        type=int,   default=512)
#     p.add_argument("--batch_size",     type=int,   default=4)
#     p.add_argument("--epochs",         type=int,   default=20)
#     p.add_argument("--lr",             type=float, default=2e-4)
#     p.add_argument("--n_local_layers", type=int,   default=1)
#     return p.parse_args()


# # ── Main ─────────────────────────────────────────────────────────────────────

# def main():
#     args = parse_args()
#     os.makedirs(args.save_dir, exist_ok=True)

#     # ── Data ──────────────────────────────────────────────────────────────
#     print("Loading CoNLL-U data...")
#     sentences = read_conll(args.data)
#     all_tags  = sorted({t for _, tags in sentences for t in tags})
#     tag2id    = {t: i for i, t in enumerate(all_tags)}
#     n_tags    = len(tag2id)
#     print(f"  Sentences: {len(sentences)}  |  Tags ({n_tags}): {all_tags}")
#     torch.save(tag2id, os.path.join(args.save_dir, "tag2id.pt"))

#     train_ds = ChunkPOSDataset(sentences, tag2id, args.max_len, "train")
#     val_ds   = ChunkPOSDataset(sentences, tag2id, args.max_len, "val")
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size,
#                               shuffle=True,  collate_fn=chunk_collate_fn, num_workers=0)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
#                               shuffle=False, collate_fn=chunk_collate_fn, num_workers=0)
#     print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

#     # ── Encoder ───────────────────────────────────────────────────────────
#     print("Loading H-Net encoder...")
#     encoder, _ = load_hnet_encoder(
#         checkpoint_path = args.hnet_ckpt,
#         model_id        = MODEL_ID,
#         device          = str(DEVICE),
#         dtype           = DTYPE,
#         frozen          = True,
#         load_lm_head    = False,
#         n_local_layers  = args.n_local_layers,
#     )
#     encoder.eval()
#     d_model = encoder.byte_embedding.embedding_dim
#     print(f"  Encoder d_model: {d_model}")

#     # ── Chunk POS head ────────────────────────────────────────────────────
#     pos_head = ChunkPOSHead(d_model, n_tags).to(DEVICE, dtype=DTYPE)
#     n_params = sum(p.numel() for p in pos_head.parameters())
#     print(f"  POS head params: {n_params:,}")

#     # ── Optimizer ─────────────────────────────────────────────────────────
#     optimizer   = torch.optim.AdamW(pos_head.parameters(), lr=args.lr, weight_decay=0.01)
#     total_steps = len(train_loader) * args.epochs
#     scheduler   = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps   = max(1, total_steps // 20),
#         num_training_steps = total_steps,
#     )

#     # ── Training loop ─────────────────────────────────────────────────────
#     best_val_acc = 0.0
#     history      = []

#     print(f"\nTraining chunk-level POS probe for {args.epochs} epochs...")
#     print("  Comparison: byte-level frozen probe = 32.2%\n")

#     for epoch in range(1, args.epochs + 1):
#         pos_head.train()
#         train_loss, train_acc, n_batches = 0.0, 0.0, 0

#         for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
#             input_ids      = batch["input_ids"].to(DEVICE)
#             attention_mask = batch["attention_mask"].to(DEVICE)
#             word_ids       = batch["word_ids"].to(DEVICE)
#             word_tags      = batch["word_tags"].to(DEVICE)

#             # Get chunk embeddings (word-level representations)
#             chunk_embs_batch, chunk_ids_batch = get_chunk_embs(
#                 encoder, input_ids, attention_mask
#             )

#             # Align chunks to words → get chunk-level labels
#             chunk_labels_batch = get_chunk_word_labels(
#                 chunk_ids_batch = chunk_ids_batch,
#                 word_ids_batch  = word_ids,
#                 word_tags_batch = word_tags,
#                 n_words_list    = batch["n_words"],
#             )

#             # Move chunk embs to correct device/dtype
#             chunk_embs_batch = [c.to(DEVICE, dtype=DTYPE) for c in chunk_embs_batch]

#             loss = pos_head.compute_loss(chunk_embs_batch, chunk_labels_batch)
#             acc  = pos_head.accuracy(chunk_embs_batch, chunk_labels_batch)

#             if torch.isnan(loss):
#                 optimizer.zero_grad()
#                 continue

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(pos_head.parameters(), 1.0)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()

#             train_loss += loss.item()
#             train_acc  += acc
#             n_batches  += 1

#         train_loss /= max(n_batches, 1)
#         train_acc  /= max(n_batches, 1)

#         # Validation
#         val_loss, val_acc = _eval(encoder, pos_head, val_loader)

#         print(f"Epoch {epoch:3d}/{args.epochs} | "
#               f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
#               f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

#         history.append({
#             "epoch": epoch,
#             "train_loss": train_loss, "train_acc": train_acc,
#             "val_loss":   val_loss,   "val_acc":   val_acc,
#         })
#         torch.save(history, os.path.join(args.save_dir, "loss_history.pt"))

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(
#                 {"pos_head": pos_head.state_dict(),
#                  "tag2id":   tag2id,
#                  "n_tags":   n_tags,
#                  "d_model":  d_model,
#                  "val_acc":  best_val_acc,
#                  "epoch":    epoch,
#                  "level":    "chunk"},
#                 os.path.join(args.save_dir, "best_chunk_pos_head.pt"),
#             )
#             print(f"  ✓ Saved best (val_acc={best_val_acc:.4f})")

#     print(f"\nChunk-level POS complete.")
#     print(f"  Best val accuracy  : {best_val_acc*100:.1f}%")
#     print(f"  Byte-level baseline: 32.2%")
#     gap = (best_val_acc - 0.322) * 100
#     print(f"  Chunk vs byte gap  : {gap:+.1f} pp")
#     print(f"  Saved to           : {args.save_dir}/")


# @torch.no_grad()
# def _eval(encoder, pos_head, loader):
#     encoder.eval()
#     pos_head.eval()
#     total_loss, total_acc, n = 0.0, 0.0, 0

#     for batch in loader:
#         input_ids      = batch["input_ids"].to(DEVICE)
#         attention_mask = batch["attention_mask"].to(DEVICE)
#         word_ids       = batch["word_ids"].to(DEVICE)
#         word_tags      = batch["word_tags"].to(DEVICE)

#         chunk_embs_batch, chunk_ids_batch = get_chunk_embs(
#             encoder, input_ids, attention_mask
#         )
#         chunk_labels_batch = get_chunk_word_labels(
#             chunk_ids_batch = chunk_ids_batch,
#             word_ids_batch  = word_ids,
#             word_tags_batch = word_tags,
#             n_words_list    = batch["n_words"],
#         )
#         chunk_embs_batch = [c.to(DEVICE, dtype=DTYPE) for c in chunk_embs_batch]

#         loss = pos_head.compute_loss(chunk_embs_batch, chunk_labels_batch)
#         acc  = pos_head.accuracy(chunk_embs_batch, chunk_labels_batch)

#         if not torch.isnan(loss):
#             total_loss += loss.item()
#         total_acc += acc
#         n         += 1

#     return total_loss / max(n, 1), total_acc / max(n, 1)


# if __name__ == "__main__":
#     main()

# scripts/eval_chunk_pos.py
"""
Chunk-level POS tagging evaluation.

Supports two evaluation protocols:
  1. Cross-lingual transfer (XL): train probe on 80% of --data language,
     evaluate on 20%. Optionally also evaluate on --test_bho and --test_mr
     using a fresh probe trained on 80% of that language's data.

  2. Zero-shot (ZS): train probe on 80% of --data (Hindi),
     then apply the SAME probe directly to --test_bho and --test_mr
     without any retraining.

Both protocols are reported when --test_bho or --test_mr are provided.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m scripts.eval_chunk_pos \
        --hnet_ckpt runs_qwen/hnet_pretrain_pos_guided/step_7000.pt \
        --data      data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
        --test_bho  data/ud_bhojpuri/bho_bhtb-ud-test.conllu \
        --test_mr   data/ud_marathi/mr_ufal-ud-train.conllu \
        --save_dir  runs_qwen/hnet_chunk_pos_full \
        --epochs    20 --n_local_layers 1
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from models.load_hnet import load_hnet_encoder
from configs.default  import DEVICE, DTYPE, MODEL_ID


# ── CoNLL-U Reader 

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


# ── Dataset 

class ChunkPOSDataset(Dataset):
    IGNORE_IDX = -100
    PAD_ID     = 0
    EOS_ID     = 1

    def __init__(self, sentences, tag2id, max_len=512,
                 split="train", train_ratio=0.8, seed=42):
        self.tag2id  = tag2id
        self.max_len = max_len

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
        byte_ids, word_ids, word_tags = [], [], []

        for w_idx, (word, tag) in enumerate(zip(words, tags)):
            word_bytes = list(word.encode("utf-8"))
            tag_id     = self.tag2id.get(tag, 0)
            word_tags.append(tag_id)
            for b in word_bytes:
                byte_ids.append(b)
                word_ids.append(w_idx)
            byte_ids.append(ord(" "))
            word_ids.append(-1)

        byte_ids  = byte_ids[:self.max_len - 1]
        word_ids  = word_ids[:self.max_len - 1]
        byte_ids.append(self.EOS_ID);  word_ids.append(-1)
        pad_len   = self.max_len - len(byte_ids)
        byte_ids += [self.PAD_ID] * pad_len
        word_ids += [-1]          * pad_len

        return {
            "input_ids":      torch.tensor(byte_ids, dtype=torch.long),
            "attention_mask": (torch.tensor(byte_ids, dtype=torch.long) != self.PAD_ID).long(),
            "word_ids":       torch.tensor(word_ids, dtype=torch.long),
            "word_tags":      torch.tensor(word_tags, dtype=torch.long),
            "n_words":        len(words),
        }


def chunk_collate_fn(batch):
    max_words   = max(b["n_words"] for b in batch)
    padded_tags = torch.full((len(batch), max_words), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        padded_tags[i, :b["n_words"]] = b["word_tags"]
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "word_ids":       torch.stack([b["word_ids"]        for b in batch]),
        "word_tags":      padded_tags,
        "n_words":        [b["n_words"] for b in batch],
    }


# ── Chunk-word label alignment 

def get_chunk_word_labels(chunk_ids_batch, word_ids_batch, word_tags_batch, n_words_list):
    B = len(chunk_ids_batch)
    chunk_labels_batch = []
    for i in range(B):
        chunk_ids = chunk_ids_batch[i]
        word_ids  = word_ids_batch[i]
        word_tags = word_tags_batch[i]
        T_i = len(chunk_ids)
        C_i = chunk_ids.max().item() + 1 if len(chunk_ids) > 0 else 0

        chunk_word_votes = {}
        for t in range(min(T_i, word_ids.shape[0])):
            c = chunk_ids[t].item()
            w = word_ids[t].item()
            if w >= 0 and c >= 0:
                if c not in chunk_word_votes:
                    chunk_word_votes[c] = {}
                chunk_word_votes[c][w] = chunk_word_votes[c].get(w, 0) + 1

        labels = torch.full((C_i,), -100, dtype=torch.long)
        for c, votes in chunk_word_votes.items():
            if c < C_i and votes:
                best_word = max(votes, key=votes.get)
                if best_word < word_tags.shape[0]:
                    labels[c] = word_tags[best_word]
        chunk_labels_batch.append(labels)
    return chunk_labels_batch


# ── POS Head 

class ChunkPOSHead(nn.Module):
    def __init__(self, d_model, n_tags, dropout=0.1):
        super().__init__()
        hidden          = min(256, d_model // 4)
        self.dropout    = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(d_model, hidden)
        self.act        = nn.GELU()
        self.norm       = nn.LayerNorm(hidden)
        self.proj       = nn.Linear(hidden, n_tags)

    def forward(self, chunk_embs):
        h = self.dropout(chunk_embs)
        h = self.act(self.bottleneck(h))
        h = self.norm(h)
        return self.proj(h)

    def compute_loss(self, chunk_embs_batch, chunk_labels_batch):
        total_loss, total_count = torch.tensor(0.0, device=chunk_embs_batch[0].device), 0
        for embs, labels in zip(chunk_embs_batch, chunk_labels_batch):
            labels = labels.to(embs.device)
            mask   = labels != -100
            if mask.sum() == 0: continue
            logits = self.forward(embs)
            loss   = F.cross_entropy(logits[mask], labels[mask], reduction="sum")
            total_loss  = total_loss + loss
            total_count += mask.sum().item()
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def accuracy(self, chunk_embs_batch, chunk_labels_batch):
        correct, total = 0, 0
        for embs, labels in zip(chunk_embs_batch, chunk_labels_batch):
            labels = labels.to(embs.device)
            mask   = labels != -100
            if mask.sum() == 0: continue
            preds   = self.forward(embs).argmax(-1)
            correct += (preds[mask] == labels[mask]).sum().item()
            total   += mask.sum().item()
        return correct / total if total > 0 else 0.0


# ── Encoder forward 

@torch.no_grad()
def get_chunk_embs(encoder, input_ids, attention_mask):
    h_hat, p, aux  = encoder(input_ids, attention_mask)
    chunk_out       = aux["chunk_out"]
    chunk_embs      = chunk_out.get("chunk_embs_smooth", chunk_out["chunk_embs"])
    chunk_ids       = chunk_out["chunk_ids"]
    return chunk_embs, chunk_ids


# ── Eval loop 

@torch.no_grad()
def _eval(encoder, pos_head, loader):
    encoder.eval(); pos_head.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        word_ids       = batch["word_ids"].to(DEVICE)
        word_tags      = batch["word_tags"].to(DEVICE)
        chunk_embs_batch, chunk_ids_batch = get_chunk_embs(encoder, input_ids, attention_mask)
        chunk_labels_batch = get_chunk_word_labels(
            chunk_ids_batch, word_ids, word_tags, batch["n_words"])
        chunk_embs_batch = [c.to(DEVICE, dtype=DTYPE) for c in chunk_embs_batch]
        loss = pos_head.compute_loss(chunk_embs_batch, chunk_labels_batch)
        acc  = pos_head.accuracy(chunk_embs_batch, chunk_labels_batch)
        if not torch.isnan(loss): total_loss += loss.item()
        total_acc += acc; n += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


# ── Train probe helper 

def train_probe(encoder, pos_head, train_loader, val_loader, args, desc=""):
    """Train probe, return best val accuracy."""
    optimizer = torch.optim.AdamW(pos_head.parameters(),
                                  lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
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
            word_ids       = batch["word_ids"].to(DEVICE)
            word_tags      = batch["word_tags"].to(DEVICE)

            chunk_embs_batch, chunk_ids_batch = get_chunk_embs(
                encoder, input_ids, attention_mask)
            chunk_labels_batch = get_chunk_word_labels(
                chunk_ids_batch, word_ids, word_tags, batch["n_words"])
            chunk_embs_batch = [c.to(DEVICE, dtype=DTYPE) for c in chunk_embs_batch]

            loss = pos_head.compute_loss(chunk_embs_batch, chunk_labels_batch)
            acc  = pos_head.accuracy(chunk_embs_batch, chunk_labels_batch)
            if torch.isnan(loss): optimizer.zero_grad(); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pos_head.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            train_loss += loss.item(); train_acc += acc; n_batches += 1

        train_loss /= max(n_batches, 1)
        train_acc  /= max(n_batches, 1)
        val_loss, val_acc = _eval(encoder, pos_head, val_loader)
        print(f"  {desc} Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f}/{train_acc:.4f} | "
              f"val={val_loss:.4f}/{val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc


# ── Argparse 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hnet_ckpt",      required=True)
    p.add_argument("--data",           required=True,
                   help="Primary language CoNLL-U (Hindi for training probe)")
    p.add_argument("--test_bho",       default=None,
                   help="Bhojpuri CoNLL-U for cross-lingual eval")
    p.add_argument("--test_mr",        default=None,
                   help="Marathi CoNLL-U for cross-lingual eval")
    p.add_argument("--save_dir",       default="runs/hnet_chunk_pos")
    p.add_argument("--max_len",        type=int,   default=512)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--n_local_layers", type=int,   default=1)
    return p.parse_args()


# ── Main 

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading primary language data...")
    sentences = read_conll(args.data)
    all_tags  = sorted({t for _, tags in sentences for t in tags})
    tag2id    = {t: i for i, t in enumerate(all_tags)}
    n_tags    = len(tag2id)
    print(f"  Sentences: {len(sentences)}  |  Tags ({n_tags}): {all_tags}")
    torch.save(tag2id, os.path.join(args.save_dir, "tag2id.pt"))

    train_ds = ChunkPOSDataset(sentences, tag2id, args.max_len, "train",
                               train_ratio=0.8, seed=42)
    val_ds   = ChunkPOSDataset(sentences, tag2id, args.max_len, "val",
                               train_ratio=0.8, seed=42)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=chunk_collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=chunk_collate_fn,
                              num_workers=0)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Encoder ───────────────────────────────────────────────────────────
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

    # ── Train probe on primary language (Hindi) ────────────────────────────
    print(f"\n{'='*60}")
    print(f"PRIMARY LANGUAGE PROBE TRAINING")
    print(f"{'='*60}")
    pos_head = ChunkPOSHead(d_model, n_tags).to(DEVICE, dtype=DTYPE)
    print(f"  POS head params: {sum(p.numel() for p in pos_head.parameters()):,}")

    best_primary_acc = train_probe(
        encoder, pos_head, train_loader, val_loader, args, desc="Primary"
    )
    print(f"\n  Primary language best val accuracy: {best_primary_acc*100:.1f}%")

    # Save best probe state for zero-shot evaluation
    torch.save(
        {"pos_head": pos_head.state_dict(),
         "tag2id": tag2id, "n_tags": n_tags,
         "d_model": d_model, "val_acc": best_primary_acc},
        os.path.join(args.save_dir, "best_chunk_pos_head.pt"),
    )

    results = {"primary": best_primary_acc}

    # ── Cross-lingual evaluation ───────────────────────────────────────────
    # For each target language we report TWO numbers:
    #   ZS: zero-shot — use Hindi-trained probe directly, no retraining
    #   XL: cross-lingual transfer — train fresh probe on 80% target data

    def eval_language(lang_name, lang_path, hindi_probe):
        print(f"\n{'='*60}")
        print(f"{lang_name.upper()} EVALUATION")
        print(f"{'='*60}")

        lang_sentences = read_conll(lang_path)
        print(f"  Total sentences: {len(lang_sentences)}")

        # ── Zero-shot (ZS): Hindi probe → all target data ─────────────────
        print(f"\n  [ZS] Zero-shot: Hindi-trained probe on all {lang_name} data")
        zs_ds = ChunkPOSDataset(
            lang_sentences, tag2id, args.max_len,
            split="val", train_ratio=0.0, seed=42  # all data as test
        )
        zs_loader = DataLoader(zs_ds, batch_size=args.batch_size,
                               shuffle=False, collate_fn=chunk_collate_fn,
                               num_workers=0)
        _, zs_acc = _eval(encoder, hindi_probe, zs_loader)
        print(f"  [ZS] {lang_name} accuracy: {zs_acc*100:.1f}%")

        # ── Cross-lingual (XL): fresh probe trained on 80% target data ────
        print(f"\n  [XL] Cross-lingual: fresh probe trained on 80% {lang_name} data")
        xl_train_ds = ChunkPOSDataset(
            lang_sentences, tag2id, args.max_len,
            split="train", train_ratio=0.8, seed=42
        )
        xl_val_ds = ChunkPOSDataset(
            lang_sentences, tag2id, args.max_len,
            split="val", train_ratio=0.8, seed=42
        )
        xl_train_loader = DataLoader(xl_train_ds, batch_size=args.batch_size,
                                     shuffle=True,  collate_fn=chunk_collate_fn,
                                     num_workers=0)
        xl_val_loader   = DataLoader(xl_val_ds,   batch_size=args.batch_size,
                                     shuffle=False, collate_fn=chunk_collate_fn,
                                     num_workers=0)
        print(f"  Train: {len(xl_train_ds)} | Val: {len(xl_val_ds)}")

        fresh_probe = ChunkPOSHead(d_model, n_tags).to(DEVICE, dtype=DTYPE)
        xl_acc = train_probe(
            encoder, fresh_probe, xl_train_loader, xl_val_loader,
            args, desc=lang_name
        )
        print(f"  [XL] {lang_name} best val accuracy: {xl_acc*100:.1f}%")

        return zs_acc, xl_acc

    # Run Bhojpuri
    if args.test_bho and os.path.exists(args.test_bho):
        zs_bho, xl_bho = eval_language("Bhojpuri", args.test_bho, pos_head)
        results["Bhojpuri_ZS"] = zs_bho
        results["Bhojpuri_XL"] = xl_bho

    # Run Marathi
    if args.test_mr and os.path.exists(args.test_mr):
        zs_mr, xl_mr = eval_language("Marathi", args.test_mr, pos_head)
        results["Marathi_ZS"] = zs_mr
        results["Marathi_XL"] = xl_mr

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — H-Net Chunk-Level POS")
    print(f"{'='*60}")
    print(f"  Encoder: frozen, pretrained on Hindi only")
    print(f"  ZS = zero-shot (Hindi probe, no target language training)")
    print(f"  XL = cross-lingual transfer (80% target language training)")
    print()
    print(f"{'Language':12s} | {'Protocol':8s} | {'Accuracy':10s} | {'#Sentences':12s}")
    print("-"*50)
    sizes = {"Primary": len(sentences), "Bhojpuri": 357, "Marathi": 500}

    print(f"{'Hindi':12s} | {'XL':8s} | {best_primary_acc*100:6.1f}%    | {len(sentences):,}")

    if "Bhojpuri_ZS" in results:
        print(f"{'Bhojpuri':12s} | {'ZS':8s} | {results['Bhojpuri_ZS']*100:6.1f}%    | 357")
    if "Bhojpuri_XL" in results:
        print(f"{'Bhojpuri':12s} | {'XL':8s} | {results['Bhojpuri_XL']*100:6.1f}%    | 357")
    if "Marathi_ZS" in results:
        print(f"{'Marathi':12s}  | {'ZS':8s} | {results['Marathi_ZS']*100:6.1f}%    | ~500")
    if "Marathi_XL" in results:
        print(f"{'Marathi':12s}  | {'XL':8s} | {results['Marathi_XL']*100:6.1f}%    | ~500")

    torch.save(results, os.path.join(args.save_dir, "results.pt"))
    print(f"\nSaved to: {args.save_dir}/results.pt")


if __name__ == "__main__":
    main()