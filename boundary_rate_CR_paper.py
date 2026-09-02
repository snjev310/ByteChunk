import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('figures', exist_ok=True)

def smooth(values, window=200):
    if len(values) < window:
        return np.array(values), np.arange(len(values))
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')
    steps = np.arange(window//2, len(values) - window//2 + 1)
    return smoothed, steps

# ── Backbone init ──────────────────────────────────────────
h1 = torch.load('runs_qwen/hnet_pretrain_pos_guided/loss_history.pt',
                map_location='cpu')
steps_b  = np.array(h1['step'])
bratio_b = np.array(h1['avg_boundary_ratio'])
bratio_b_s, steps_b_s = smooth(bratio_b, 200)

mask = steps_b_s <= 25000
steps_b_s  = steps_b_s[mask]
bratio_b_s = bratio_b_s[mask]

# ── Random init ───────────────────────────────────────────
h2 = torch.load("/home/sanjeev/work/hnet-2/runs/hnet_en/loss_history.pt",
                map_location='cpu')
steps_r  = np.array(h2['step']) * 4   # global steps
bratio_r = np.array(h2["p_mean"])

mask_r = steps_r <= 25000
steps_r  = steps_r[mask_r]
bratio_r = bratio_r[mask_r]
bratio_r_s, steps_r_s = smooth(bratio_r, 200)

# ── Plot ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.grid":        True,
    "grid.color":       "#dddddd",
    "grid.linewidth":   0.6,
    "grid.linestyle":   "--",
    "font.size":        12,
})

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(steps_r_s, bratio_r_s, color='tomato',
        linewidth=2.0, label='Random initialisation')
ax.plot(steps_b_s, bratio_b_s, color='royalblue',
        linewidth=2.0, label='Backbone initialisation')
ax.axhline(0.056, color='black', linestyle='--',
           linewidth=1.2, label=r'Target $\rho^*=0.056$')

ax.set_xlabel('Training step', fontsize=12)
ax.set_ylabel('Boundary ratio', fontsize=12)
ax.set_title('Boundary ratio during H-Net pretraining', fontsize=13)
ax.set_xlim(0, 25000)
ax.set_ylim(0, 0.6)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('figures/boundary_ratio_comparison.png',
            dpi=150, bbox_inches='tight')
print('Saved: figures/boundary_ratio_comparison.png')