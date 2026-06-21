"""
FlyLLM - Adaptive Quantizer
Compresses layer files into ~/flyllmmodel/ModelName/
Only saves model.layers.{idx}.safetensors + profile.json
Static files (embed, norm, lm_head) stay in HF cache.
"""

import os
import json
import torch
from typing import Optional
from safetensors.torch import load_file, save_file

from .config import ModelConfig, get_flyllm_dir

BLOCK_SIZE = 32


# ── Quantize one tensor ───────────────────────────────────────

def quantize_tensor(tensor: torch.Tensor, precision: str):
    t = tensor.float()

    if precision == "float16" or t.dim() < 2:
        return tensor.half(), None

    orig  = t.shape
    flat  = t.flatten()
    n     = flat.numel()
    pad   = (BLOCK_SIZE - n % BLOCK_SIZE) % BLOCK_SIZE
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])

    blocks = flat.reshape(-1, BLOCK_SIZE)

    if precision == "int8":
        scales = blocks.abs().max(dim=1).values / 127.0
        scales = scales.clamp(min=1e-8)
        q      = (blocks / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    else:  # int4
        scales = blocks.abs().max(dim=1).values / 7.0
        scales = scales.clamp(min=1e-8)
        q      = (blocks / scales.unsqueeze(1)).round().clamp(-8, 7).to(torch.int8)

    meta = {
        "scales": scales.half(),
        "shape":  torch.tensor(list(orig) + [pad], dtype=torch.int32),
        "prec":   torch.tensor(
            [{"float16": 0, "int8": 1, "int4": 2}[precision]], dtype=torch.int32
        ),
    }
    return q, meta


def dequantize_tensor(q: torch.Tensor, meta: Optional[dict],
                      dtype=torch.float16) -> torch.Tensor:
    if meta is None:
        return q.to(dtype)
    scales     = meta["scales"].float()
    shape_data = meta["shape"].tolist()
    orig_shape, pad = shape_data[:-1], shape_data[-1]
    flat = (q.float() * scales.unsqueeze(1)).flatten()
    if pad:
        flat = flat[:-pad]
    return flat.reshape(orig_shape).to(dtype)


# ── Quantize one layer file ───────────────────────────────────

def _quantize_and_save_layer(src_path: str, dst_path: str, precision: str):
    tensors = load_file(src_path, device="cpu")
    out     = {}

    for key, tensor in tensors.items():
        q, meta = quantize_tensor(tensor, precision)
        out[key] = q.contiguous()
        if meta:
            out[f"{key}.__scales"] = meta["scales"].contiguous()
            out[f"{key}.__shape"]  = meta["shape"]
            out[f"{key}.__prec"]   = meta["prec"]

    save_file(out, dst_path)
    return os.path.getsize(src_path), os.path.getsize(dst_path)


# ── Main quantize ─────────────────────────────────────────────

def quantize_model(
    model_id:  str,
    hf_dir:    str,
    profile:   dict,
    output_dir: Optional[str] = None,
    verbose:   bool = True,
) -> str:
    """
    Quantize all layer files from hf_dir into output_dir.
    Only saves model.layers.{idx}.safetensors files.
    Static files (embed, norm, lm_head) stay in HF cache.

    Args:
        model_id:   HuggingFace model ID
        hf_dir:     path to HF splitted_model directory
        profile:    precision profile from profiler
        output_dir: where to save (default: ~/flyllmmodel/ModelName)
        verbose:    print progress

    Returns:
        path to output directory
    """
    if output_dir is None:
        output_dir = get_flyllm_dir(model_id)

    os.makedirs(output_dir, exist_ok=True)

    cfg = profile.get("summary", {})
    num_layers = len(profile["layers"])

    if verbose:
        print(f"\n{'='*65}")
        print(f"  FlyLLM - Quantizing {num_layers} layers")
        print(f"  Output: {output_dir}")
        print(f"{'='*65}\n")

    total_orig = total_quant = 0

    for idx in range(num_layers):
        precision = profile["layers"].get(str(idx), {}).get("precision", "int4")
        src       = os.path.join(hf_dir, f"model.layers.{idx}.safetensors")
        dst       = os.path.join(output_dir, f"model.layers.{idx}.safetensors")

        if not os.path.exists(src):
            continue

        orig_size, quant_size = _quantize_and_save_layer(src, dst, precision)
        total_orig  += orig_size
        total_quant += quant_size

        if verbose:
            icon  = {"float16": "🔴", "int8": "🟡", "int4": "🟢"}[precision]
            ratio = orig_size / max(quant_size, 1)
            print(f"  L{idx:<4} {icon} {precision:<8}  "
                  f"{orig_size/1e6:.0f}MB → {quant_size/1e6:.0f}MB  ({ratio:.1f}x)")

    # Save profile alongside layers
    profile_path = os.path.join(output_dir, "flyllm_profile.json")
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    # Save metadata
    meta = {
        "flyllm_version": "1.0",
        "model_id":       model_id,
        "model_type":     profile.get("model_type", "unknown"),
        "hf_dir":         hf_dir,          # ← remember where original is
        "avg_bits":       profile["summary"]["avg_bits"],
        "size_reduction": profile["summary"]["size_reduction"],
        "float16_layers": profile["summary"]["float16_layers"],
        "int8_layers":    profile["summary"]["int8_layers"],
        "int4_layers":    profile["summary"]["int4_layers"],
    }
    with open(os.path.join(output_dir, "flyllm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    reduction = round((1 - total_quant / max(total_orig, 1)) * 100, 1)

    if verbose:
        print(f"\n  Original : {total_orig/1e9:.2f} GB")
        print(f"  FlyLLM   : {total_quant/1e9:.2f} GB")
        print(f"  Reduction: {reduction}%")
        print(f"  Saved to : {output_dir}\n")

    return output_dir
