"""
FlyLLM - Layer Sensitivity Profiler
Exactly like combined_profiler.py that proved the method.
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

# Thresholds — proven on Mistral 7B
THRESHOLD_FLOAT16 = 0.35
THRESHOLD_INT8    = 0.15


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


def _score_layer(tensors: dict) -> dict:
    """Compute combined score for one layer — exact same logic as combined_profiler.py"""
    all_vals = []
    for tensor in tensors.values():
        if tensor.dim() >= 2:
            all_vals.append(tensor.float().flatten())

    if not all_vals:
        return {"kurtosis": 3.0, "entropy": 0.5, "max_abs": 0.1,
                "kurt_norm": 0.0, "entr_norm": 0.5, "mabs_norm": 0.0,
                "score": 0.1, "precision": "int4"}

    flat = torch.cat(all_vals)

    kurt = _kurtosis(flat)
    entr = _entropy(flat)
    mabs = _max_abs(flat)

    # Normalize — same ranges as combined_profiler.py
    kurt_norm = float(np.clip((kurt - 3.0) / 17.0, 0, 1))
    mabs_norm = float(np.clip((mabs - 0.05) / 0.45, 0, 1))
    entr_norm = entr  # already normalized to [0,1]

    # Weighted score — 0.5 / 0.3 / 0.2
    score = (kurt_norm * 0.5) + (entr_norm * 0.3) + (mabs_norm * 0.2)

    precision = (
        "float16" if score >= THRESHOLD_FLOAT16 else
        "int8"    if score >= THRESHOLD_INT8    else
        "int4"
    )

    return {
        "kurtosis":  round(kurt, 4),
        "entropy":   round(entr, 4),
        "max_abs":   round(mabs, 4),
        "kurt_norm": round(kurt_norm, 4),
        "entr_norm": round(entr_norm, 4),
        "mabs_norm": round(mabs_norm, 4),
        "score":     round(score, 4),
        "precision": precision,
    }


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
    all_scores = []

    for idx in range(cfg.num_hidden_layers):
        path = os.path.join(hf_dir, f"model.layers.{idx}.safetensors")
        if not os.path.exists(path):
            continue

        tensors = load_file(path, device="cpu")
        result  = _score_layer(tensors)
        del tensors

        profile["layers"][str(idx)] = result
        counts[result["precision"]] += 1
        total_bits += {"float16": 16, "int8": 8, "int4": 4}[result["precision"]]
        all_scores.append(result["score"])

        if verbose:
            bar_len = int(result["score"] * 22)
            bar     = "█" * bar_len + "░" * (22 - bar_len)
            icon    = {"float16": "★", "int8": "◆", "int4": "·"}[result["precision"]]
            print(f"  L{idx:<4} {result['kurtosis']:>8.3f} {result['entropy']:>8.3f} "
                  f"{result['max_abs']:>8.3f} {result['score']:>8.4f}  {bar}  "
                  f"{icon} {result['precision']}")

    avg_bits  = total_bits / cfg.num_hidden_layers
    reduction = round((1 - avg_bits / 16) * 100, 1)

    profile["summary"] = {
        "avg_bits":       round(avg_bits, 2),
        "size_reduction": reduction,
        "float16_layers": counts["float16"],
        "int8_layers":    counts["int8"],
        "int4_layers":    counts["int4"],
        "score_min":      round(min(all_scores), 4),
        "score_max":      round(max(all_scores), 4),
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"  PROFILE SUMMARY")
        print(f"{'='*80}")
        print(f"  float16 : {counts['float16']:2d} layers  ★ (score >= {THRESHOLD_FLOAT16} — critical)")
        print(f"  int8    : {counts['int8']:2d} layers  ◆ (score >= {THRESHOLD_INT8} — moderate)")
        print(f"  int4    : {counts['int4']:2d} layers  · (score <  {THRESHOLD_INT8} — compressible)")
        print(f"\n  Score range : {min(all_scores):.4f} → {max(all_scores):.4f}")
        print(f"  Avg bits    : {avg_bits:.2f}")
        print(f"  Reduction   : {reduction}%")
        print(f"{'='*80}\n")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)
        if verbose:
            print(f"  Profile saved → {output_path}\n")

    return profile
