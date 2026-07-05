"""
FlyLLM - Layer Sensitivity Profiler
3 metrics: Kurtosis + Entropy + MaxAbs
Zero calibration data needed.

Normalization strategy:
Each raw metric (kurtosis, max_abs) is min-max normalized ACROSS THIS
MODEL'S OWN LAYERS before being combined into a score. No fixed/hardcoded
ranges are used — those broke on non-Mistral models because a range tuned
on one model's weight statistics doesn't transfer to another architecture.

Threshold strategy:
Instead of comparing each layer's score to a fixed constant (which only
means the same thing across models if their score distributions happen
to have the same shape — they don't), thresholds are found PER MODEL by
locating the largest natural gaps in that model's own sorted score
distribution. This is the numpy-native equivalent of 1D k-means with
k=3, without the dependency or runtime cost: with only ~24-32 layers,
the largest gap in sorted scores IS the natural clean split. This avoids
forcing a fixed proportion of layers into each precision tier (which
percentile thresholds would do) and avoids assuming a fixed score value
means the same thing on every architecture (which fixed thresholds did,
and which was the original bug).
"""

import os
import json
import torch
import numpy as np
from typing import Optional
from safetensors.torch import load_file

from .config import ModelConfig, load_config, get_hf_cache_dir, get_flyllm_dir


# Fallback constants only — used if a model's scores are too uniform
# to find a meaningful gap (degenerate case, should be rare in practice).
THRESHOLD_FLOAT16 = 0.35
THRESHOLD_INT8    = 0.18


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
    # Normalize: max entropy = log(32). Bounded by construction (a
    # histogram entropy has a known max), so it's already comparable
    # across models — no cross-layer min-max needed for this one.
    return float(np.clip(entr / np.log(32), 0, 1))


def _max_abs(flat: torch.Tensor) -> float:
    return float(flat.abs().max())


def _raw_metrics(tensors: dict) -> dict:
    """Compute raw (un-normalized) metrics for one layer."""
    all_vals = []
    for tensor in tensors.values():
        if tensor.dim() >= 2:
            all_vals.append(tensor.float().flatten())

    if not all_vals:
        # Fallback for empty/unexpected layer — neutral-ish values,
        # will land near the middle after min-max.
        return {"kurtosis": 3.0, "entropy": 0.5, "max_abs": 0.05}

    flat = torch.cat(all_vals)
    return {
        "kurtosis": _kurtosis(flat),
        "entropy":  _entropy(flat),
        "max_abs":  _max_abs(flat),
    }


def _minmax(values: list) -> list:
    """Min-max normalize a list of raw values to [0, 1], per model."""
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1e-8  # guard: flat model, avoid div/0
    return [float(np.clip((v - lo) / span, 0, 1)) for v in values]


def _find_thresholds(scores: list, verbose: bool = True) -> tuple:
    """
    Find natural breakpoints for THIS model by locating the largest gaps
    in the sorted score distribution. Pure numpy, no clustering library,
    no fixed proportion of layers forced into any tier.

    Returns (float16_cut, int8_cut).
    """
    arr = np.sort(np.array(scores))

    if arr[-1] - arr[0] < 1e-6:
        if verbose:
            print(f"  [thresholds] scores too uniform to split — "
                  f"falling back to fixed defaults ({THRESHOLD_FLOAT16}/{THRESHOLD_INT8})")
        return THRESHOLD_FLOAT16, THRESHOLD_INT8

    gaps = np.diff(arr)  # gap between each consecutive sorted pair

    # Largest gap overall -> splits int4 (below) from everything else (above)
    idx1 = int(np.argmax(gaps))
    int8_cut = float((arr[idx1] + arr[idx1 + 1]) / 2)
    gap1_size = float(gaps[idx1])

    # Largest gap ABOVE that cut -> splits remaining into int8 vs float16
    upper = arr[idx1 + 1:]
    if len(upper) > 1:
        upper_gaps = np.diff(upper)
        idx2 = int(np.argmax(upper_gaps))
        float16_cut = float((upper[idx2] + upper[idx2 + 1]) / 2)
        gap2_size = float(upper_gaps[idx2])
    else:
        float16_cut = float(arr[-1] + 1e-6)  # nothing left to split -> no float16 tier
        gap2_size = 0.0

    if verbose:
        print(f"  [thresholds] int8_cut={int8_cut:.4f} (gap={gap1_size:.4f})   "
              f"float16_cut={float16_cut:.4f} (gap={gap2_size:.4f})")
        # A tiny gap size relative to the score range means the "split" may
        # just be noise on a smooth gradient rather than a real cluster
        # boundary — worth eyeballing when gap sizes look small.
        score_range = arr[-1] - arr[0]
        if gap1_size < 0.05 * score_range or gap2_size < 0.05 * score_range:
            print(f"  [thresholds] WARNING: at least one gap is small relative to "
                  f"the score range ({score_range:.4f}) — this model's sensitivity "
                  f"may be a smooth gradient rather than clean clusters. Inspect "
                  f"the printed scores below before trusting this split.")

    return float16_cut, int8_cut


def profile_model(
    model_id:    str,
    hf_dir:      str,
    output_path: Optional[str] = None,
    verbose:     bool = True,
) -> dict:
    """
    Profile all layers in the HF splitted_model directory.
    Saves profile.json inside the flyllm model directory.

    Args:
        model_id:    HuggingFace model ID
        hf_dir:      path to splitted_model in HF cache
        output_path: where to save profile.json
        verbose:     print progress table
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"  FlyLLM - Layer Sensitivity Profiler")
        print(f"  Model  : {model_id}")
        print(f"  Source : {hf_dir}")
        print(f"{'='*80}\n")

    cfg = load_config(model_id)

    if verbose:
        print(f"  Architecture : {cfg.model_type}")
        print(f"  Layers       : {cfg.num_hidden_layers}\n")

    profile = {
        "model_id":   model_id,
        "model_type": cfg.model_type,
        "layers":     {},
        "summary":    {},
    }

    # ---- Pass 1: compute raw metrics for every layer ----
    raw_by_idx = {}
    for idx in range(cfg.num_hidden_layers):
        path = os.path.join(hf_dir, f"model.layers.{idx}.safetensors")
        if not os.path.exists(path):
            continue
        tensors = load_file(path, device="cpu")
        raw_by_idx[idx] = _raw_metrics(tensors)
        del tensors

    if not raw_by_idx:
        raise RuntimeError(f"No layer files found in {hf_dir}")

    indices = sorted(raw_by_idx.keys())
    kurts   = [raw_by_idx[i]["kurtosis"] for i in indices]
    entrs   = [raw_by_idx[i]["entropy"]  for i in indices]  # already [0,1]
    mabss   = [raw_by_idx[i]["max_abs"]  for i in indices]

    # ---- Min-max normalize kurtosis and max_abs across THIS model's layers ----
    kurt_norms = _minmax(kurts)
    mabs_norms = _minmax(mabss)
    entr_norms = entrs  # bounded by construction, no min-max needed

    # ---- Pass 2: combine into scores for every layer ----
    scores_by_pos = []
    for pos in range(len(indices)):
        score = (kurt_norms[pos] * 0.5) + (entr_norms[pos] * 0.3) + (mabs_norms[pos] * 0.2)
        scores_by_pos.append(score)

    # ---- Find this model's own natural thresholds from its score distribution ----
    float16_cut, int8_cut = _find_thresholds(scores_by_pos, verbose=verbose)

    # ---- Pass 3: assign precision using the per-model cutoffs, print ----
    if verbose:
        print(f"\n  {'L':<5} {'Kurt':>8} {'Entrop':>8} {'MaxAbs':>8} "
              f"{'SCORE':>8}  {'Bar':<22} Precision")
        print(f"  {'-'*75}")

    counts = {"float16": 0, "int8": 0, "int4": 0}
    total_bits = 0
    all_scores = []

    for pos, idx in enumerate(indices):
        raw = raw_by_idx[idx]
        score = scores_by_pos[pos]

        precision = (
            "float16" if score >= float16_cut else
            "int8"    if score >= int8_cut    else
            "int4"
        )

        result = {
            "kurtosis":  round(raw["kurtosis"], 4),
            "entropy":   round(raw["entropy"], 4),
            "max_abs":   round(raw["max_abs"], 4),
            "kurt_norm": round(kurt_norms[pos], 4),
            "entr_norm": round(entr_norms[pos], 4),
            "mabs_norm": round(mabs_norms[pos], 4),
            "score":     round(score, 4),
            "precision": precision,
        }

        profile["layers"][str(idx)] = result
        counts[precision] += 1
        total_bits += {"float16": 16, "int8": 8, "int4": 4}[precision]
        all_scores.append(score)

        if verbose:
            bar_len = int(score * 22)
            bar     = "█" * bar_len + "░" * (22 - bar_len)
            icon    = {"float16": "★", "int8": "◆", "int4": "·"}[precision]
            print(f"  L{idx:<4} {result['kurtosis']:>8.3f} {result['entropy']:>8.3f} "
                  f"{result['max_abs']:>8.3f} {result['score']:>8.4f}  {bar}  "
                  f"{icon} {precision}")

    avg_bits  = total_bits / len(indices)
    reduction = round((1 - avg_bits / 16) * 100, 1)

    profile["summary"] = {
        "avg_bits":       round(avg_bits, 2),
        "size_reduction": reduction,
        "float16_layers": counts["float16"],
        "int8_layers":    counts["int8"],
        "int4_layers":    counts["int4"],
        "score_min":      round(min(all_scores), 4),
        "score_max":      round(max(all_scores), 4),
        "float16_cut":    round(float16_cut, 4),
        "int8_cut":       round(int8_cut, 4),
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"  PROFILE SUMMARY")
        print(f"{'='*80}")
        print(f"  float16 : {counts['float16']:2d} layers  ★ (score >= {float16_cut:.4f} — critical)")
        print(f"  int8    : {counts['int8']:2d} layers  ◆ (score >= {int8_cut:.4f} — moderate)")
        print(f"  int4    : {counts['int4']:2d} layers  · (score <  {int8_cut:.4f} — compressible)")
        print(f"\n  Score range          : {min(all_scores):.4f} → {max(all_scores):.4f}")
        print(f"  Cutoffs (this model) : int8={int8_cut:.4f}  float16={float16_cut:.4f}")
        print(f"  Avg bits             : {avg_bits:.2f}")
        print(f"  Reduction            : {reduction}%")
        print(f"{'='*80}\n")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)
        if verbose:
            print(f"  Profile saved → {output_path}\n")

    return profile
