# scripts/eval_llama_pos.py
"""
Evaluate LLaMA 3.1-8B (with LoRA) on Hindi POS tagging using subword tokens.
Compares subword-level LLaMA representations vs byte-level H-Net representations.

Pipeline:
  1. Load LLaMA + LoRA (same config as H-Net pretraining)
  2. Tokenize Hindi words using LLaMA tokenizer (subword)
  3. Extract last hidden state for each word (use first subword token)
  4. Train linear POS head on top of frozen LLaMA representations
  5. Report accuracy — compare with H-Net byte-level (32.2%)

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m scripts.eval_llama_pos \
        --hnet_ckpt runs_v2/hnet_pretrain/best.pt \
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
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

from configs.default import DEVICE, DTYPE, MODEL_ID


# ── Dataset ───────────────────────────────────────────────────────────────────

class SubwordPOSDataset(Dataset):
    """
    Subword-level POS dataset for LLaMA.
    Each sentence is tokenized with the LLaMA tokenizer.
    Labels are assigned to the FIRST subword token of each word.
    All other subword tokens get IGNORE_IDX=-100.
    Max sequence length capped at 512 tokens.
    """

    IGNORE_IDX = -100

    def __init__(self, sentences, tokenizer, tag2id, max_len=512,
                 split="train", train_ratio=0.8, seed=42):
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

        input_ids = []
        labels    = []

        for word, tag in zip(words, tags):
            # Tokenize word — prepend space for proper subword splitting
            word_tokens = self.tokenizer.encode(
                " " + word, add_special_tokens=False
            )
            if not word_tokens:
                continue

            tag_id = self.tag2id.get(tag, 0)

            # First subword token gets the label, rest get IGNORE
            for j, tok in enumerate(word_tokens):
                input_ids.append(tok)
                labels.append(tag_id if j == 0 else self.IGNORE_IDX)

        # Add BOS
        bos = self.tokenizer.bos_token_id or 1
        input_ids = [bos] + input_ids
        labels    = [self.IGNORE_IDX] + labels

        # Truncate
        input_ids = input_ids[:self.max_len]
        labels    = labels[:self.max_len]

        # Pad
        pad_id  = self.tokenizer.pad_token_id or 0
        pad_len = self.max_len - len(input_ids)
        input_ids += [pad_id]  * pad_len
        labels    += [self.IGNORE_IDX] * pad_len

        input_ids_t      = torch.tensor(input_ids, dtype=torch.long)
        attention_mask   = (input_ids_t != pad_id).long()
        labels_t         = torch.tensor(labels,    dtype=torch.long)

        return {
            "input_ids":      input_ids_t,
            "attention_mask": attention_mask,
            "labels":         labels_t,
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── POS Head ─────────────────────────────────────────────────────────────────

class SubwordPOSHead(nn.Module):
    """
    Linear probe on top of LLaMA hidden states.
    Uses same bottleneck as byte-level POSHead for fair comparison.
    d_model=4096 → 256 → n_tags
    """
    def __init__(self, d_model: int, n_tags: int, dropout: float = 0.1):
        super().__init__()
        hidden          = min(256, d_model // 4)
        self.dropout    = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(d_model, hidden, bias=True)
        self.act        = nn.GELU()
        self.norm       = nn.LayerNorm(hidden)
        self.proj       = nn.Linear(hidden, n_tags, bias=True)

    def forward(self, h):
        h = self.dropout(h)
        h = self.act(self.bottleneck(h))
        h = self.norm(h)
        return self.proj(h)

    def compute_loss(self, h, labels):
        logits = self.forward(h)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def word_accuracy(self, h, labels):
        preds   = self.forward(h).argmax(-1).reshape(-1)
        targets = labels.reshape(-1)
        mask    = targets != -100
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == targets[mask]).sum().item() / mask.sum().item()


# ── CoNLL-U Reader ────────────────────────────────────────────────────────────

def read_conll(path):
    sentences, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "" or line.startswith("#"):
                if words:
                    sentences.append((words, tags))
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
    p.add_argument("--hnet_ckpt", required=True,
                   help="H-Net checkpoint (to load LoRA weights from)")
    p.add_argument("--data",      required=True)
    p.add_argument("--save_dir",  default="runs/llama_pos")
    p.add_argument("--max_len",   type=int,   default=512)
    p.add_argument("--batch_size",type=int,   default=4)
    p.add_argument("--epochs",    type=int,   default=20)
    p.add_argument("--lr",        type=float, default=2e-4)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading CoNLL-U data...")
    sentences = read_conll(args.data)
    all_tags  = sorted({tag for _, tags in sentences for tag in tags})
    tag2id    = {tag: i for i, tag in enumerate(all_tags)}
    n_tags    = len(tag2id)
    print(f"  Sentences : {len(sentences)}")
    print(f"  POS tags  : {n_tags}  {all_tags}")

    # ── 2. Load LLaMA + LoRA (same config as H-Net) ───────────────────────
    print(f"\nLoading LLaMA backbone: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map="auto",
    )
    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(base, lora_config)

    # Load LoRA weights from H-Net checkpoint
    print(f"  Loading LoRA weights from: {args.hnet_ckpt}")
    ckpt       = torch.load(args.hnet_ckpt, map_location=str(DEVICE))
    lora_state = ckpt.get("lora_state", {})
    if lora_state:
        missing, unexpected = peft_model.load_state_dict(lora_state, strict=False)
        print(f"  LoRA loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("  Warning: no LoRA weights found — using base LLaMA")

    # Freeze all LLaMA parameters
    for param in peft_model.parameters():
        param.requires_grad = False
    peft_model.eval()
    print("  LLaMA frozen")

    hidden_size = peft_model.config.hidden_size
    print(f"  Hidden size: {hidden_size}")

    # ── 3. Build datasets ─────────────────────────────────────────────────
    print("\nBuilding datasets...")
    train_ds = SubwordPOSDataset(sentences, tokenizer, tag2id,
                                  args.max_len, split="train")
    val_ds   = SubwordPOSDataset(sentences, tokenizer, tag2id,
                                  args.max_len, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 4. POS head ───────────────────────────────────────────────────────
    pos_head = SubwordPOSHead(hidden_size, n_tags).to(DEVICE, dtype=DTYPE)
    n_params = sum(p.numel() for p in pos_head.parameters())
    print(f"  POS head params: {n_params:,}")

    # ── 5. Optimizer ──────────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(pos_head.parameters(),
                                     lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, total_steps // 20),
        num_training_steps = total_steps,
    )

    # ── 6. Training loop ──────────────────────────────────────────────────
    best_val_acc = 0.0
    history      = []

    print(f"\nTraining LLaMA subword POS probe for {args.epochs} epochs...")
    print("  This is the SUBWORD baseline for comparison with H-Net byte-level (32.2%)\n")

    for epoch in range(1, args.epochs + 1):
        pos_head.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            # Get LLaMA hidden states — frozen, no grad
            with torch.no_grad():
                out    = peft_model.model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    output_hidden_states = True,
                    return_dict    = True,
                    use_cache      = False,
                )
                # Last hidden state: [B, T, 4096]
                h = out.hidden_states[-1]

            loss = pos_head.compute_loss(h, labels)
            acc  = pos_head.word_accuracy(h, labels)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pos_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_acc  += acc
            n_batches  += 1

        train_loss = train_loss / max(n_batches, 1)
        train_acc  = train_acc  / max(n_batches, 1)

        # Validation
        val_loss, val_acc = _eval(peft_model, pos_head, val_loader)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
        })
        torch.save(history, os.path.join(args.save_dir, "loss_history.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"pos_head": pos_head.state_dict(),
                 "tag2id":   tag2id,
                 "n_tags":   n_tags,
                 "val_acc":  best_val_acc,
                 "epoch":    epoch,
                 "mode":     "subword_llama"},
                os.path.join(args.save_dir, "best_pos_head.pt"),
            )
            print(f"  ✓ Saved best (val_acc={best_val_acc:.4f})")

    # ── 7. Final comparison ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUBWORD vs BYTE-LEVEL POS COMPARISON")
    print("="*60)
    print(f"  LLaMA subword  (this run) : {best_val_acc*100:.1f}%")
    print(f"  H-Net byte-level (frozen) : 32.2%")
    print(f"  Random baseline           :  6.25%")
    gap = (best_val_acc - 0.322) * 100
    print(f"  Subword - Byte gap        : {gap:+.1f} pp")
    print("="*60)
    print(f"\nSaved to: {args.save_dir}/")


@torch.no_grad()
def _eval(llama_model, pos_head, loader):
    pos_head.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        out = llama_model.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True, use_cache=False,
        )
        h    = out.hidden_states[-1]
        loss = pos_head.compute_loss(h, labels)
        acc  = pos_head.word_accuracy(h, labels)
        if not torch.isnan(loss):
            total_loss += loss.item()
        total_acc  += acc
        n          += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


if __name__ == "__main__":
    main()