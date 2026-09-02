# utils/logging.py
import os
import torch


def init_loss_history():
    return {
        "step":               [],
        "total_loss":         [],
        "loss_ar":            [],
        "loss_ratio":         [],
        "loss_align":         [],
        "bpb":                [],
        "avg_boundary_ratio": [],
        "p_mean":             [],
        "p_std":              [],
        "p_min":              [],
        "p_max":              [],
        "avg_num_chunks":     [],
        "avg_chunk_length":   [],
    }


def update_loss_history(
    history: dict,
    *,
    step:       int,
    total:      float,
    ar:         float,
    ratio:      float,
    align:      float,
    bpb:        float,
    boundary:   float,
    p_stats:    dict = None,
    chunk_stats: dict = None,
):
    history["step"].append(step)
    history["total_loss"].append(total)
    history["loss_ar"].append(ar)
    history["loss_ratio"].append(ratio)
    history["loss_align"].append(align)
    history["bpb"].append(bpb)
    history["avg_boundary_ratio"].append(boundary)

    if p_stats:
        history["p_mean"].append(p_stats["mean"])
        history["p_std"].append(p_stats["std"])
        history["p_min"].append(p_stats["min"])
        history["p_max"].append(p_stats["max"])

    if chunk_stats:
        history["avg_num_chunks"].append(chunk_stats["avg_num_chunks"])
        history["avg_chunk_length"].append(chunk_stats["avg_chunk_length"])


def save_loss_history(history: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    tmp   = os.path.join(save_dir, "loss_history.tmp.pt")
    final = os.path.join(save_dir, "loss_history.pt")
    torch.save(history, tmp)
    os.replace(tmp, final)
