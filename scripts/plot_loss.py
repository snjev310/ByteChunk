# scripts/plot_loss.py
"""
Plot training metrics from loss_history.pt

Usage:
    python -m scripts.plot_loss --history runs/hnet_pretrain/loss_history.pt
    python -m scripts.plot_loss --history runs/hnet_pretrain/loss_history.pt --save plots.png
"""

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def smooth(values, window=10):
    """Simple moving average for noisy curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_metrics(history: dict, save_path: str = None, smooth_window: int = 10):

    steps = history["step"]

    # ── Determine what's available ────────────────────────────────────────
    has_total    = len(history.get("total_loss", [])) > 0
    has_ar       = len(history.get("loss_ar", [])) > 0
    has_bpb      = len(history.get("bpb", [])) > 0
    has_ratio    = len(history.get("loss_ratio", [])) > 0
    has_boundary = len(history.get("avg_boundary_ratio", [])) > 0
    has_p_mean   = len(history.get("p_mean", [])) > 0
    has_chunks   = len(history.get("avg_chunk_length", [])) > 0

    n_plots = sum([
        has_total or has_ar,
        has_bpb,
        has_boundary or has_p_mean,
        has_chunks or has_ratio,
    ])
    n_plots = max(n_plots, 1)

    fig = plt.figure(figsize=(14, 4 * n_plots))
    fig.suptitle("H-Net Training Metrics", fontsize=15, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(n_plots, 1, hspace=0.45)

    plot_idx = 0

    # ── 1. Loss curves ────────────────────────────────────────────────────
    if has_total or has_ar:
        ax = fig.add_subplot(gs[plot_idx]); plot_idx += 1
        ax.set_title("Loss", fontweight="bold")

        if has_total:
            vals = history["total_loss"]
            ax.plot(steps, vals, alpha=0.25, color="steelblue", linewidth=0.8)
            s = smooth(vals, smooth_window)
            ax.plot(steps[len(steps)-len(s):], s, color="steelblue",
                    linewidth=2, label="total loss")

        if has_ar:
            vals = history["loss_ar"]
            ax.plot(steps, vals, alpha=0.25, color="darkorange", linewidth=0.8)
            s = smooth(vals, smooth_window)
            ax.plot(steps[len(steps)-len(s):], s, color="darkorange",
                    linewidth=2, label="AR loss")

        if has_ratio:
            vals = history["loss_ratio"]
            ax.plot(steps, vals, alpha=0.25, color="green", linewidth=0.8)
            s = smooth(vals, smooth_window)
            ax.plot(steps[len(steps)-len(s):], s, color="green",
                    linewidth=2, label="ratio loss")

        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Annotate final value
        if has_ar:
            final = history["loss_ar"][-1]
            ax.annotate(f"final AR: {final:.3f}",
                        xy=(steps[-1], final),
                        xytext=(-60, 10), textcoords="offset points",
                        fontsize=9, color="darkorange",
                        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1))

    # ── 2. Bits per byte ──────────────────────────────────────────────────
    if has_bpb:
        ax = fig.add_subplot(gs[plot_idx]); plot_idx += 1
        ax.set_title("Bits per Byte (BPB)  ↓ lower is better", fontweight="bold")

        vals = history["bpb"]
        ax.plot(steps, vals, alpha=0.25, color="purple", linewidth=0.8)
        s = smooth(vals, smooth_window)
        ax.plot(steps[len(steps)-len(s):], s, color="purple",
                linewidth=2, label="BPB")

        # Reference lines
        ax.axhline(y=1.0, color="green",  linestyle="--", alpha=0.5, label="BPB=1.0 (excellent)")
        # ax.axhline(y=2.0, color="orange", linestyle="--", alpha=0.5, label="BPB=2.0 (good)")
        ax.axhline(y=8.0, color="red",    linestyle="--", alpha=0.4, label="BPB=8.0 (random)")

        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("BPB")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        final_bpb = history["bpb"][-1]
        ax.annotate(f"final: {final_bpb:.3f}",
                    xy=(steps[-1], final_bpb),
                    xytext=(-60, 10), textcoords="offset points",
                    fontsize=9, color="purple",
                    arrowprops=dict(arrowstyle="->", color="purple", lw=1))

    # ── 3. Boundary ratio + p stats ───────────────────────────────────────
    if has_boundary or has_p_mean:
        ax = fig.add_subplot(gs[plot_idx]); plot_idx += 1
        ax.set_title("Boundary Rate  (target = 0.056 = 1 boundary per 18 bytes)",
                     fontweight="bold")

        if has_boundary:
            vals = history["avg_boundary_ratio"]
            ax.plot(steps, vals, alpha=0.25, color="crimson", linewidth=0.8)
            s = smooth(vals, smooth_window)
            ax.plot(steps[len(steps)-len(s):], s, color="crimson",
                    linewidth=2, label="boundary rate")

        if has_p_mean:
            vals = history["p_mean"]
            ax.plot(steps, vals, alpha=0.25, color="teal", linewidth=0.8)
            s = smooth(vals, smooth_window)
            ax.plot(steps[len(steps)-len(s):], s, color="teal",
                    linewidth=2, label="p mean", linestyle="--")

        # Target reference line
        ax.axhline(y=0.056, color="black", linestyle="--",
                   alpha=0.6, linewidth=1.5, label="target (0.056)")
        ax.axhline(y=0.5,   color="gray",  linestyle=":",
                   alpha=0.4, linewidth=1,   label="random init (0.5)")

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Rate")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        if has_boundary:
            final_br = history["avg_boundary_ratio"][-1]
            color = "green" if abs(final_br - 0.056) < 0.05 else "red"
            ax.annotate(f"final: {final_br:.3f}",
                        xy=(steps[-1], final_br),
                        xytext=(-60, 10), textcoords="offset points",
                        fontsize=9, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1))

    # ── 4. Chunk length + ratio loss ──────────────────────────────────────
    if has_chunks:
        ax = fig.add_subplot(gs[plot_idx]); plot_idx += 1
        ax.set_title("Avg Chunk Length  (target ~16 bytes/chunk)", fontweight="bold")

        vals = history["avg_chunk_length"]
        ax.plot(steps, vals, alpha=0.25, color="saddlebrown", linewidth=0.8)
        s = smooth(vals, smooth_window)
        ax.plot(steps[len(steps)-len(s):], s, color="saddlebrown",
                linewidth=2, label="avg chunk length")

        ax.axhline(y=8.0,  color="black",  linestyle="--",
                   alpha=0.6, linewidth=1.5, label="target (16 bytes)")
        ax.axhline(y=4.5,  color="green",  linestyle=":",
                   alpha=0.5, linewidth=1,   label="paper range (4.5–8)")

        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Bytes / chunk")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
    else:
        plt.show()

    # ── Print summary table ───────────────────────────────────────────────
    print("\n── Training Summary ──────────────────────────────────────")
    print(f"  Total optimizer steps : {len(steps)}")
    if has_total:
        print(f"  Total loss  start → end : {history['total_loss'][0]:.4f} → {history['total_loss'][-1]:.4f}")
    if has_ar:
        print(f"  AR loss     start → end : {history['loss_ar'][0]:.4f} → {history['loss_ar'][-1]:.4f}")
    if has_bpb:
        print(f"  BPB         start → end : {history['bpb'][0]:.4f} → {history['bpb'][-1]:.4f}")
    if has_boundary:
        br_final = history["avg_boundary_ratio"][-1]
        status   = "✓ on target" if abs(br_final - 0.056) < 0.05 else "✗ off target"
        print(f"  Boundary rate (final)   : {br_final:.4f}  {status}")
    if has_chunks:
        print(f"  Avg chunk len (final)   : {history['avg_chunk_length'][-1]:.2f} bytes")
    print("──────────────────────────────────────────────────────────")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True, help="Path to loss_history.pt")
    p.add_argument("--save",    default=None,  help="Save plot to this path (e.g. plots.png)")
    p.add_argument("--smooth",  type=int, default=10, help="Smoothing window (default 10)")
    args = p.parse_args()

    print(f"Loading: {args.history}")
    history = torch.load(args.history, map_location="cpu")

    print(f"  Keys     : {list(history.keys())}")
    print(f"  Steps    : {len(history.get('step', []))}")

    plot_metrics(history, save_path=args.save, smooth_window=args.smooth)


if __name__ == "__main__":
    main()


