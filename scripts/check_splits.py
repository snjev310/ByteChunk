# scripts/check_splits.py
"""
Compute word, subword, and chunk statistics for all evaluation languages.
Reports:
  - Number of sentences and words
  - Qwen subword tokens per word (SW/W)
  - H-Net chunks per word (C/W)
  - Percentage of chunks covering exactly one CoNLL-U word (1:1 align%)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.check_splits
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scripts.eval_chunk_pos import (
    read_conll, ChunkPOSDataset, chunk_collate_fn, get_chunk_embs
)
from models.load_hnet import load_hnet_encoder
from configs.default import DEVICE, DTYPE, MODEL_ID


# ── Config ────────────────────────────────────────────────────────────────────

HNET_CKPT    = "runs_qwen/hnet_pretrain_pos_guided/step_7000.pt"
N_LOCAL_LAYERS = 1

DATASETS = {
    "Hindi":    "data/ud_hindi_treebank/hi_hdtb-ud-train.conllu",
    "Bhojpuri": "data/ud_bhojpuri/bho_bhtb-ud-test.conllu",
    "Marathi":  "data/ud_marathi/mr_ufal-ud-train.conllu",
    "Magahi":   "data/ud_magahi/mag_mgtb-ud-test.conllu",
    "Sanskrit": "data/ud_sanskrit/sa_ufal-ud-test.conllu",
    "Urdu":     "data/ud_urdu/ur_udtb-ud-test.conllu",
}


# ── Load models once ──────────────────────────────────────────────────────────

def load_models():
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading H-Net encoder...")
    encoder, _ = load_hnet_encoder(
        checkpoint_path = HNET_CKPT,
        model_id        = MODEL_ID,
        device          = str(DEVICE),
        dtype           = DTYPE,
        frozen          = True,
        load_lm_head    = False,
        n_local_layers  = N_LOCAL_LAYERS,
    )
    encoder.eval()
    print()
    return tokenizer, encoder


# ── Per-language stats ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_stats(lang, path, tokenizer, encoder):
    # Read CoNLL data
    sents  = read_conll(path)
    tag2id = {t: i for i, t in enumerate(
        sorted({t for _, tags in sents for t in tags}))}

    # Dataset — use ALL sentences as test (train_ratio=0.0)
    ds = ChunkPOSDataset(
        sents, tag2id, max_len=512,
        split="val", train_ratio=0.0, seed=42
    )
    loader = DataLoader(
        ds, batch_size=4,
        shuffle=False, collate_fn=chunk_collate_fn, num_workers=0
    )

    total_sents   = len(sents)
    total_words   = 0
    total_subwords = 0
    total_chunks  = 0
    perfect_align = 0   # chunks covering exactly 1 word
    total_chunks_counted = 0

    # Track which sentence we are on for word lookup
    sent_idx = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        word_ids_batch = batch["word_ids"]      # [B, T] on CPU
        n_words_list   = batch["n_words"]

        chunk_embs_batch, chunk_ids_batch = get_chunk_embs(
            encoder, input_ids, attention_mask)

        for i, (chunk_ids, n_words) in enumerate(
                zip(chunk_ids_batch, n_words_list)):

            # Words in this sentence
            if sent_idx < len(sents):
                words, _ = sents[sent_idx]
            else:
                words = []
            sent_idx += 1

            total_words += n_words

            # Count subword tokens for each word
            for word in words[:n_words]:
                toks = tokenizer.encode(
                    " " + word, add_special_tokens=False)
                total_subwords += max(len(toks), 1)

            # Count chunks
            n_chunks = int(chunk_ids.max().item()) + 1
            total_chunks += n_chunks

            # Count 1:1 chunk-word alignments
            wid_row = word_ids_batch[i]   # [T]

            for c in range(n_chunks):
                words_in_chunk = set()
                for t in range(min(len(chunk_ids), wid_row.shape[0])):
                    if chunk_ids[t].item() == c:
                        w = wid_row[t].item()
                        if w >= 0:
                            words_in_chunk.add(w)

                total_chunks_counted += 1
                if len(words_in_chunk) == 1:
                    perfect_align += 1

    sw_per_word  = total_subwords / max(total_words, 1)
    c_per_word   = total_chunks   / max(total_words, 1)
    align_pct    = perfect_align  / max(total_chunks_counted, 1) * 100

    return {
        "lang":          lang,
        "sents":         total_sents,
        "words":         total_words,
        "subwords":      total_subwords,
        "chunks":        total_chunks,
        "sw_per_word":   sw_per_word,
        "c_per_word":    c_per_word,
        "align_pct":     align_pct,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    tokenizer, encoder = load_models()

    results = []
    for lang, path in DATASETS.items():
        print(f"Processing {lang}...")
        stats = compute_stats(lang, path, tokenizer, encoder)
        results.append(stats)

    # ── Print table ───────────────────────────────────────────────────────
    print()
    print("=" * 88)
    print("WORD / SUBWORD / CHUNK STATISTICS")
    print("=" * 88)
    print(f"{'Language':12s} | {'Sents':6s} | {'Words':8s} | "
          f"{'Subwords':10s} | {'Chunks':8s} | "
          f"{'SW/W':6s} | {'C/W':6s} | {'1:1 align%':10s}")
    print("-" * 88)

    for r in results:
        print(f"{r['lang']:12s} | {r['sents']:6,d} | {r['words']:8,d} | "
              f"{r['subwords']:10,d} | {r['chunks']:8,d} | "
              f"{r['sw_per_word']:6.2f} | {r['c_per_word']:6.2f} | "
              f"{r['align_pct']:8.1f}%")

    # ── Print LaTeX table ─────────────────────────────────────────────────
    print()
    print("=" * 88)
    print("LATEX TABLE")
    print("=" * 88)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"\textbf{Language} & \textbf{Words} & \textbf{SW/W} & "
          r"\textbf{C/W} & \textbf{1:1 Align\%} \\")
    print(r"\midrule")
    for r in results:
        print(f"{r['lang']:12s} & {r['words']:,d} & "
              f"{r['sw_per_word']:.2f} & "
              f"{r['c_per_word']:.2f} & "
              f"{r['align_pct']:.1f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Word, subword, and chunk statistics. "
          r"SW/W = Qwen subword tokens per word (higher = more "
          r"fragmentation). C/W = H-Net chunks per word (ideally "
          r"$\approx$1.0). 1:1 Align\% = percentage of chunks "
          r"covering exactly one CoNLL-U word.}")
    print(r"\label{tab:alignment}")
    print(r"\end{table}")

    # ── Summary for paper text ────────────────────────────────────────────
    print()
    print("=" * 88)
    print("SUMMARY FOR PAPER")
    print("=" * 88)
    min_align = min(r["align_pct"] for r in results)
    max_align = max(r["align_pct"] for r in results)
    min_cw    = min(r["c_per_word"] for r in results)
    max_cw    = max(r["c_per_word"] for r in results)
    print(f"1:1 alignment range: {min_align:.1f}% -- {max_align:.1f}%")
    print(f"Chunks per word range: {min_cw:.2f} -- {max_cw:.2f}")
    print()
    print("Paper sentence:")
    print(f"  H-Net chunks align with CoNLL-U words in "
          f"{min_align:.0f}--{max_align:.0f}\\% of cases across all "
          f"six languages (C/W = {min_cw:.2f}--{max_cw:.2f}), "
          f"validating chunk-level evaluation as a word-level protocol.")


if __name__ == "__main__":
    main()