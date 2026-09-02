# scripts/plot_mt_loss.py
"""
Plot train/val loss and BPB for encoder-decoder MT training.

Usage:
    # Live — run while training is in progress
    python -m scripts.plot_mt_loss --history runs/hnet_encdec_en_hi/loss_history.pt

    # Save to file
    python -m scripts.plot_mt_loss --history runs/hnet_encdec_en_hi/loss_history.pt --save mt_plot.png
"""

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(history: dict, save_path: str = None):
    epochs     = history["epoch"]
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    train_bpb  = history["train_bpb"]
    val_bpb    = history["val_bpb"]

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle("H-Net Enc-Dec MT Training\n(Hindi → Bhojpuri)",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── 1. Loss ───────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])   # full width
    ax.set_title("Cross-Entropy Loss", fontweight="bold")
    ax.plot(epochs, train_loss, "o-", color="steelblue",  linewidth=2,
            markersize=5, label="Train loss")
    ax.plot(epochs, val_loss,   "s-", color="darkorange", linewidth=2,
            markersize=5, label="Val loss")

    # Annotate best val
    best_epoch = epochs[val_loss.index(min(val_loss))]
    best_val   = min(val_loss)
    ax.axvline(x=best_epoch, color="darkorange", linestyle="--",
               alpha=0.5, linewidth=1)
    ax.annotate(f"best val\n{best_val:.4f}\n(epoch {best_epoch})",
                xy=(best_epoch, best_val),
                xytext=(10, 10), textcoords="offset points",
                fontsize=8, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 2. BPB ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title("Bits per Byte (BPB)  ↓ lower is better", fontweight="bold")
    ax2.plot(epochs, train_bpb, "o-", color="purple",   linewidth=2,
             markersize=5, label="Train BPB")
    ax2.plot(epochs, val_bpb,   "s-", color="crimson",  linewidth=2,
             markersize=5, label="Val BPB")

    # Reference lines
    ax2.axhline(y=8.0, color="red",    linestyle=":", alpha=0.4, label="8.0 (random)")
    ax2.axhline(y=4.0, color="orange", linestyle=":", alpha=0.4, label="4.0 (partial)")
    ax2.axhline(y=2.0, color="green",  linestyle=":", alpha=0.4, label="2.0 (good)")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BPB")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Summary box ───────────────────────────────────────────────────────
    last_epoch  = epochs[-1]
    summary = (
        f"Epochs completed : {last_epoch}\n"
        f"Best val loss    : {best_val:.4f}  (epoch {best_epoch})\n"
        f"Best val BPB     : {min(val_bpb):.3f}\n"
        f"Train loss (last): {train_loss[-1]:.4f}\n"
        f"Val loss (last)  : {val_loss[-1]:.4f}\n"
        f"Gap (overfit?)   : {val_loss[-1] - train_loss[-1]:.4f}"
    )
    fig.text(0.5, -0.02, summary, ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
             family="monospace")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    # Console summary
    print("\n── MT Training Summary ───────────────────────────────")
    print(f"  Epochs done   : {last_epoch}")
    print(f"  Best val loss : {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Best val BPB  : {min(val_bpb):.3f}")
    print(f"  Final train   : {train_loss[-1]:.4f}  ({train_bpb[-1]:.3f} BPB)")
    print(f"  Final val     : {val_loss[-1]:.4f}  ({val_bpb[-1]:.3f} BPB)")
    gap = val_loss[-1] - train_loss[-1]
    overfitting = "possible overfitting" if gap > 0.3 else "no overfitting"
    print(f"  Train/val gap : {gap:.4f}  ({overfitting})")

    # Convergence check
    if len(val_bpb) >= 5:
        last5 = val_bpb[-5:]
        improvement = last5[0] - last5[-1]
        if improvement < 0.05:
            print(f"  Convergence   : ⚠ plateaued (only {improvement:.4f} BPB improvement in last 5 epochs)")
        else:
            print(f"  Convergence   : ✓ still improving ({improvement:.4f} BPB drop in last 5 epochs)")
    print("──────────────────────────────────────────────────────")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True, help="Path to loss_history.pt")
    p.add_argument("--save",    default=None,  help="Save plot to file (e.g. plot.png)")
    args = p.parse_args()

    history = torch.load(args.history, map_location="cpu")
    print(f"Loaded: {args.history}")
    print(f"  Epochs recorded: {len(history['epoch'])}")
    plot(history, save_path=args.save)


if __name__ == "__main__":
    main()