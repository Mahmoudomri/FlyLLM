"""
FlyLLM - Layer Sensitivity Profiler
3 metrics: Kurtosis + Entropy + MaxAbs
Zero calibration data needed.

"""

import os
import json
import torch
import numpy as np
from typing import Optional
from safetensors.torch import load_file

from .config import ModelConfig, load_config, get_hf_cache_dir, get_flyllm_dir


# ---------------------------------------------------------------------------
# Raw metric computation (unchanged)
# ---------------------------------------------------------------------------

def _kurtosis(flat: torch.Tensor) -> float:
    mean = flat.mean()
    std  = flat.std()
    if std < 1e-8:
        return 3.0
    return float((((flat - mean) / std) ** 4).mean())


def _entropy(flat: torch.Tensor) -> float:
    hist = torch.histc(flat, bins=32)
    hist = hist / (hist.sum() + 1e-8)
    entr = float(-(hist * (hist + 1e-8).log()).sum())
    # Normalize: max entropy = log(32)
    return float(np.clip(entr / np.log(32), 0, 1))


def _max_abs(flat: torch.Tensor) -> float:
    return float(flat.abs().max())


def _raw_layer_stats(tensors: dict) -> dict:
    """Compute raw kurt/entropy/max_abs + weighted SCORE for one layer.
    """
    all_vals = []
    for tensor in tensors.values():
        if tensor.dim() >= 2:
            all_vals.append(tensor.float().flatten())

    if not all_vals:
        return {"kurtosis": 3.0, "entropy": 0.5, "max_abs": 0.1,
                "kurt_norm": 0.0, "entr_norm": 0.5, "mabs_norm": 0.0,
                "score": 0.1}

    flat = torch.cat(all_vals)

    kurt = _kurtosis(flat)
    entr = _entropy(flat)
    mabs = _max_abs(flat)

    # Normalize — same ranges as before
    kurt_norm = float(np.clip((kurt - 3.0) / 17.0, 0, 1))
    mabs_norm = float(np.clip((mabs - 0.05) / 0.45, 0, 1))
    entr_norm = entr  # already normalized to [0,1]

    # Weighted score — 0.5 / 0.3 / 0.2 (kurtosis dominates on purpose:
    # it's the metric that tracks activation-outlier severity, which is
    # exactly what destroys accuracy under low-bit quantization)
    score = (kurt_norm * 0.5) + (entr_norm * 0.3) + (mabs_norm * 0.2)

    return {
        "kurtosis":  round(kurt, 4),
        "entropy":   round(entr, 4),
        "max_abs":   round(mabs, 4),
        "kurt_norm": round(kurt_norm, 4),
        "entr_norm": round(entr_norm, 4),
        "mabs_norm": round(mabs_norm, 4),
        "score":     round(score, 4),
    }


# ---------------------------------------------------------------------------
# Adaptive tiering — the new part
# ---------------------------------------------------------------------------

def _robust_z(scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Median/MAD z-score. Immune to the single-huge-outlier problem that
    breaks mean/std (one crazy layer won't drag everyone else's z down)."""
    median_s = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median_s)))
    scaled_mad = 1.4826 * mad  # convention: makes MAD ~ std under normality
    return median_s, mad, scaled_mad


def _kmeans_1d(values: np.ndarray, k: int, n_iter: int = 200) -> tuple:
    """
    Plain 1D k-means (Lloyd's algorithm).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    k = min(k, n)
    if k <= 1:
        return np.zeros(n, dtype=int), np.array([values.mean()])

    sorted_vals = np.sort(values)
    init_idx = np.linspace(0, n - 1, k).round().astype(int)
    centroids = sorted_vals[init_idx].astype(float)
    centroids = np.unique(centroids)
    while len(centroids) < k:
        centroids = np.append(centroids, centroids[-1] + 1e-6 * (len(centroids) + 1))

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists = np.abs(values[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        new_centroids = centroids.copy()
        for c in range(len(centroids)):
            mask = labels == c
            if mask.any():
                new_centroids[c] = values[mask].mean()
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    return labels, centroids


def assign_precisions(scores: list) -> list:
    """
    Fully self-calibrating precision assignment via k-means.
    """
    arr = np.array(scores, dtype=float)
    n = len(arr)

    if n == 0:
        return []
    if n == 1:
        return ["float16"]
    if n == 2:
        # Not enough points to form 3 meaningful clusters — split by rank.
        order = np.argsort(-arr)
        out = ["int4"] * n
        out[order[0]] = "float16"
        return out

    labels, centroids = _kmeans_1d(arr, k=3)

    present_clusters = sorted(set(labels.tolist()), key=lambda c: -centroids[c])
    tier_names = ["float16", "int8", "int4"]
    tier_map = {
        c: (tier_names[i] if i < len(tier_names) else "int4")
        for i, c in enumerate(present_clusters)
    }

    return [tier_map[label] for label in labels]


# ---------------------------------------------------------------------------
# Main profiling entry point
# ---------------------------------------------------------------------------

def profile_model(
    model_id:    str,
    hf_dir:      str,
    output_path: Optional[str] = None,
    verbose:     bool = True,
) -> dict:
    """
    Profile all layers in the HF splitted_model directory.
    Saves profile.json inside the flyllm model directory.

    Precision tiers (float16/int8/int4) are computed adaptively per model —
    no fixed SCORE thresholds anywhere in this function.
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"  FlyLLM - Layer Sensitivity Profiler (adaptive tiering)")
        print(f"  Model  : {model_id}")
        print(f"  Source : {hf_dir}")
        print(f"{'='*80}\n")

    cfg = load_config(model_id)

    if verbose:
        print(f"  Architecture : {cfg.model_type}")
        print(f"  Layers       : {cfg.num_hidden_layers}\n")
        print(f"  Pass 1/2 : scoring layers...")

    # ---- Pass 1: compute raw stats + score for every layer -----------------
    layer_indices = []
    raw_results   = []

    for idx in range(cfg.num_hidden_layers):
        path = os.path.join(hf_dir, f"model.layers.{idx}.safetensors")
        if not os.path.exists(path):
            continue

        tensors = load_file(path, device="cpu")
        result  = _raw_layer_stats(tensors)
        del tensors

        layer_indices.append(idx)
        raw_results.append(result)

    if not raw_results:
        raise RuntimeError(f"No layer safetensors found under {hf_dir}")

    all_scores = [r["score"] for r in raw_results]

    # ---- Pass 2: adaptive precision assignment ------------------------------
    n_layers = len(layer_indices)

    if n_layers <= 2:
        # Too few layers to meaningfully separate edges from a "middle" —
        # fall back to clustering everything together as before.
        precisions = assign_precisions(all_scores)
    else:
        middle_positions = list(range(1, n_layers - 1))
        middle_scores    = [all_scores[i] for i in middle_positions]

        middle_precisions = assign_precisions(middle_scores)

        precisions = ["float16"] * n_layers
        for local_i, pos in enumerate(middle_positions):
            precisions[pos] = middle_precisions[local_i]
        # precisions[0] and precisions[-1] stay "float16" — set above,
        # never touched by the clustering.

    if verbose:
        median_s, mad, scaled_mad = _robust_z(np.array(all_scores))
        print(f"  Pass 2/2 : adaptive tiering "
              f"(median={median_s:.4f}, scaled_MAD={scaled_mad:.4f})")
        if n_layers > 2:
            print(f"             L0 and L{n_layers-1} forced to float16, "
                  f"excluded from k-means (edge-outlier rule)\n")
        else:
            print()
        print(f"  {'L':<5} {'Kurt':>8} {'Entrop':>8} {'MaxAbs':>8} "
              f"{'SCORE':>8}  {'Bar':<22} Precision")
        print(f"  {'-'*75}")

    profile = {
        "model_id":   model_id,
        "model_type": cfg.model_type,
        "layers":     {},
        "summary":    {},
    }

    counts = {"float16": 0, "int8": 0, "int4": 0}
    total_bits = 0

    for idx, result, precision in zip(layer_indices, raw_results, precisions):
        result = dict(result)
        result["precision"] = precision
        profile["layers"][str(idx)] = result

        counts[precision] += 1
        total_bits += {"float16": 16, "int8": 8, "int4": 4}[precision]

        if verbose:
            bar_len = int(result["score"] * 22)
            bar_len = max(0, min(22, bar_len))
            bar     = "█" * bar_len + "░" * (22 - bar_len)
            icon    = {"float16": "★", "int8": "◆", "int4": "·"}[precision]
            print(f"  L{idx:<4} {result['kurtosis']:>8.3f} {result['entropy']:>8.3f} "
                  f"{result['max_abs']:>8.3f} {result['score']:>8.4f}  {bar}  "
                  f"{icon} {precision}")

    avg_bits  = total_bits / len(layer_indices)
    reduction = round((1 - avg_bits / 16) * 100, 1)

    profile["summary"] = {
        "avg_bits":       round(avg_bits, 2),
        "size_reduction": reduction,
        "float16_layers": counts["float16"],
        "int8_layers":    counts["int8"],
        "int4_layers":    counts["int4"],
        "score_min":      round(min(all_scores), 4),
        "score_max":      round(max(all_scores), 4),
        "score_median":   round(float(np.median(all_scores)), 4),
        "tiering_method": "kmeans_1d_k3_edges_excluded",  # documents HOW tiers were picked
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"  PROFILE SUMMARY  (thresholds derived from this model only)")
        print(f"{'='*80}")
        print(f"  float16 : {counts['float16']:2d} layers  ★ critical (largest score gap, upper tier)")
        print(f"  int8    : {counts['int8']:2d} layers  ◆ moderate (secondary gap tier)")
        print(f"  int4    : {counts['int4']:2d} layers  · compressible (flat baseline / noise band)")
        print(f"\n  Score range  : {min(all_scores):.4f} → {max(all_scores):.4f}")
        print(f"  Score median : {profile['summary']['score_median']:.4f}")
        print(f"  Avg bits     : {avg_bits:.2f}")
        print(f"  Reduction    : {reduction}%")
        print(f"{'='*80}\n")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)
        if verbose:
            print(f"  Profile saved → {output_path}\n")

    return profile
