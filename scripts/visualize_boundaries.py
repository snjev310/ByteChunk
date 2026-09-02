# scripts/visualize_boundaries.py
"""
Visualize H-Net boundary predictions on example sentences.

Loads only the lightweight components from a checkpoint
(byte_embedding + routing) — no LLaMA needed.

Usage:
    python -m scripts.visualize_boundaries \
        --ckpt runs/hnet_pretrain/best.pt \
        --threshold 0.4

Checkpoint keys expected (format_version=3 from training/checkpoint.py):
    byte_embedding, routing
"""

import argparse
import torch
import torch.nn as nn

from data.byte_tokenizer import ByteTokenizer
from models.routing import RoutingModule


# ─────────────────────────────────────────────────────────────────────────────
# Default examples — add your own Hindi/Bhojpuri sentences here
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    # English
    "once upon a time in a land far, far away, there lived a wise old owl.",
    # Hindi
    "बहुत समय पहले, एक बहुत दूर देश में, एक बुद्धिमान बूढ़ा उल्लू रहता था।",
    # Bhojpuri
    "राम घरे जाता बा। ऊ बहुत नीमन आदमी बा।",
    # Code
    "def compute_loss(logits, labels): return F.cross_entropy(logits, labels)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_routing_from_checkpoint(ckpt_path: str, device: torch.device):
    """
    Load only byte_embedding + routing from a format_version=3 checkpoint.
    Does NOT require LLaMA — fast to load.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # Infer model dim from embedding weight shape
    emb_weight = ckpt["byte_embedding"]["weight"]
    model_dim  = emb_weight.shape[1]
    print(f"  checkpoint model_dim = {model_dim}")
    print(f"  global_step          = {ckpt.get('global_step', '?')}")
    print(f"  best_loss            = {ckpt.get('best_loss', '?'):.4f}" if isinstance(ckpt.get('best_loss'), float) else "")

    byte_embedding = nn.Embedding(256, model_dim, padding_idx=0).to(device)
    byte_embedding.load_state_dict(ckpt["byte_embedding"])
    byte_embedding.eval()

    routing = RoutingModule(model_dim).to(device)
    routing.load_state_dict(ckpt["routing"])
    routing.eval()

    return byte_embedding, routing, model_dim


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize(
    text:          str,
    byte_embedding: nn.Embedding,
    routing:        RoutingModule,
    tokenizer:      ByteTokenizer,
    threshold:      float,
    device:         torch.device,
):
    print("\n" + "=" * 80)
    print(f"INPUT: {repr(text[:80])}{'...' if len(text) > 80 else ''}")
    print(f"UTF-8 bytes: {len(text.encode('utf-8'))}")

    input_ids      = tokenizer.encode(text).unsqueeze(0).to(device)   # [1, T]
    attention_mask = (input_ids != tokenizer.pad_token_id).long()     # [1, T]

    with torch.no_grad():
        h           = byte_embedding(input_ids).to(dtype=routing.W_q.weight.dtype)
        routing_out = routing(h)
        p           = routing_out.p   # [1, T]

    ids_list  = input_ids[0].tolist()
    mask_list = attention_mask[0].tolist()
    p_list    = p[0].tolist()
    b_list    = [1.0 if pi > threshold else 0.0 for pi in p_list]

    # ── Boundary-segmented text ───────────────────────────────────────────
    print(f"\nCHUNKS  (threshold={threshold}, | = boundary):")
    chunk_bytes = []
    output_parts = []

    for byte_val, m, bi in zip(ids_list, mask_list, b_list):
        if m == 0:
            break
        if bi == 1.0 and chunk_bytes:
            output_parts.append("|" + bytes(chunk_bytes).decode("utf-8", errors="replace"))
            chunk_bytes = []
        chunk_bytes.append(byte_val)

    if chunk_bytes:
        output_parts.append(bytes(chunk_bytes).decode("utf-8", errors="replace"))

    print("  " + "".join(output_parts))

    # ── Per-chunk stats ───────────────────────────────────────────────────
    chunk_sizes = []
    size = 0
    for m, bi in zip(mask_list, b_list):
        if m == 0:
            break
        size += 1
        if bi == 1.0 and size > 1:
            chunk_sizes.append(size - 1)
            size = 1
    if size > 0:
        chunk_sizes.append(size)

    # ── Boundary probability heatmap (compact) ────────────────────────────
    print(f"\nP HEATMAP  (each char = 1 byte, shade = boundary prob):")
    shades = " ░▒▓█"
    line = "  "
    for byte_val, m, pi in zip(ids_list, mask_list, p_list):
        if m == 0:
            break
        shade_idx = min(int(pi * len(shades)), len(shades) - 1)
        line += shades[shade_idx]
    print(line)

    # ── Summary stats ─────────────────────────────────────────────────────
    valid_p = [pi for pi, m in zip(p_list, mask_list) if m == 1]
    valid_b = [bi for bi, m in zip(b_list, mask_list) if m == 1]
    n_valid = len(valid_p)
    n_chunks = sum(1 for b in valid_b if b == 1.0) + 1  # +1 for first chunk

    print(f"\nSTATS:")
    print(f"  valid bytes       : {n_valid}")
    print(f"  chunks formed     : {n_chunks}")
    print(f"  compression ratio : {n_valid / max(n_chunks, 1):.2f} bytes/chunk  (target ~8)")
    print(f"  boundary rate     : {sum(valid_b)/n_valid:.3f}  (target ~0.056)")
    print(f"  p  mean/min/max   : {sum(valid_p)/n_valid:.3f} / {min(valid_p):.3f} / {max(valid_p):.3f}")

    if chunk_sizes:
        avg_cs = sum(chunk_sizes) / len(chunk_sizes)
        print(f"  avg chunk size    : {avg_cs:.1f} bytes")

    # ── Routing similarity stats ──────────────────────────────────────────
    sim_list = routing_out.sim[0].tolist()
    valid_sim = [s for s, m in zip(sim_list, mask_list) if m == 1]
    print(f"  sim mean/std      : {sum(valid_sim)/len(valid_sim):.3f} / "
          f"{torch.tensor(valid_sim).std().item():.3f}")
    print(f"  (sim near 1 = similar neighbors = no boundary, near -1 = boundary)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      required=True,       help="Path to best.pt or final.pt")
    p.add_argument("--threshold", type=float, default=0.4, help="Boundary threshold (default 0.4)")
    p.add_argument("--max_len",   type=int,   default=2048)
    p.add_argument("--text",      type=str,   default=None, help="Single sentence to visualize")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    byte_embedding, routing, _ = load_routing_from_checkpoint(args.ckpt, device)

    tokenizer = ByteTokenizer(max_length=args.max_len)

    examples = [args.text] if args.text else EXAMPLES

    for text in examples:
        visualize(
            text           = text,
            byte_embedding = byte_embedding,
            routing        = routing,
            tokenizer      = tokenizer,
            threshold      = args.threshold,
            device         = device,
        )

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()