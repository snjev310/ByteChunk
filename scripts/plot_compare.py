# scripts/plot_compare.py
"""
Compare training curves of two MT runs side by side.
Useful for ablation: frozen encoder (v1) vs unfrozen encoder (v2).

Usage:
    python -m scripts.plot_compare \
        --h1  runs/hnet_encdec_hi_ang/loss_history.pt \
        --h2  runs/hnet_encdec_hi_bho_v2/loss_history.pt \
        --l1  "Frozen encoder (v1)" \
        --l2  "Unfrozen encoder (v2)" \
        --save comparison.png
"""

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h1",   required=True, help="History file 1")
    p.add_argument("--h2",   required=True, help="History file 2")
    p.add_argument("--l1",   default="Run 1", help="Label for run 1")
    p.add_argument("--l2",   default="Run 2", help="Label for run 2")
    p.add_argument("--save", default=None,   help="Save path (e.g. comparison.png)")
    return p.parse_args()


def main():
    args  = parse_args()
    h1    = torch.load(args.h1, map_location="cpu")
    h2    = torch.load(args.h2, map_location="cpu")

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("MT Training Comparison\n(Hindi → Bhojpuri)",
                 fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    colors = {
        "v1_train": "steelblue",   "v1_val": "cornflowerblue",
        "v2_train": "darkorange",  "v2_val": "sandybrown",
    }

    # ── Top left: Val loss comparison ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Val Loss", fontweight="bold")
    ax.plot(h1["epoch"], h1["val_loss"], "s-", color=colors["v1_val"],
            linewidth=2, markersize=4, label=args.l1)
    ax.plot(h2["epoch"], h2["val_loss"], "s-", color=colors["v2_val"],
            linewidth=2, markersize=4, label=args.l2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Top right: Train loss comparison ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Train Loss", fontweight="bold")
    ax.plot(h1["epoch"], h1["train_loss"], "o-", color=colors["v1_train"],
            linewidth=2, markersize=4, label=args.l1)
    ax.plot(h2["epoch"], h2["train_loss"], "o-", color=colors["v2_train"],
            linewidth=2, markersize=4, label=args.l2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Bottom left: Val BPB comparison ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Val BPB  ↓ lower is better", fontweight="bold")
    ax.plot(h1["epoch"], h1["val_bpb"], "s-", color=colors["v1_val"],
            linewidth=2, markersize=4, label=args.l1)
    ax.plot(h2["epoch"], h2["val_bpb"], "s-", color=colors["v2_val"],
            linewidth=2, markersize=4, label=args.l2)
    ax.axhline(y=2.0, color="green",  linestyle="--", alpha=0.4, label="2.0 (good)")
    ax.axhline(y=1.0, color="green",  linestyle=":",  alpha=0.4, label="1.0 (excellent)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BPB")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Bottom right: Summary table ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    def fmt(h, label):
        best_val  = min(h["val_loss"])
        best_ep   = h["epoch"][h["val_loss"].index(best_val)]
        best_bpb  = min(h["val_bpb"])
        final_gap = h["val_loss"][-1] - h["train_loss"][-1]
        return [label,
                f"{best_val:.4f}",
                f"{best_bpb:.3f}",
                f"ep {best_ep}",
                f"{final_gap:+.4f}"]

    rows   = [fmt(h1, args.l1), fmt(h2, args.l2)]
    cols   = ["Run", "Best val\nloss", "Best val\nBPB", "Best\nepoch", "Final\ngap"]
    table  = ax.table(cellText=rows, colLabels=cols,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Highlight better value in each metric column green
    for col_idx in [1, 2]:   # val loss and BPB — lower is better
        vals = [float(rows[i][col_idx]) for i in range(2)]
        best_row = vals.index(min(vals))
        table[best_row + 1, col_idx].set_facecolor("#d4edda")

    ax.set_title("Summary", fontweight="bold", pad=20)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()

    # Console output
    print("\n── Comparison Summary ────────────────────────────────")
    for h, label in [(h1, args.l1), (h2, args.l2)]:
        best_val = min(h["val_loss"])
        best_bpb = min(h["val_bpb"])
        best_ep  = h["epoch"][h["val_loss"].index(best_val)]
        print(f"  {label}")
        print(f"    Best val loss : {best_val:.4f}  (epoch {best_ep})")
        print(f"    Best val BPB  : {best_bpb:.3f}")
    print("──────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()