"""
FlyLLM Benchmark
Tests float16 original vs FlyLLM adaptive on same prompts.
Generates markdown report for GitHub README.

Run:
  python -m benchmark.run_benchmark \
    --model mistralai/Mistral-7B-v0.1
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flyllm.config import load_config, get_hf_cache_dir, get_flyllm_dir, format_prompt
from flyllm.quantizer import dequantize_tensor
from flyllm.engines.mistral import MistralEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16

PROMPTS = [
    "What is 15% of 240? Give only the number.",
    "If John is taller than Mary and Mary is taller than Sue, who is the shortest? One word.",
    "A train travels at 60km/h for 2 hours. How far? Number and unit only.",
    "All birds have wings. Penguins are birds. Do penguins have wings? Yes or no and why.",
    "What is the capital of Germany? One word.",
]

EXPECTED = ["36", "Sue", "120 km", "yes", "Berlin"]


def cosine_sim(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def measure_cosine(hf_dir, flyllm_dir, num_layers):
    """Compare cosine similarity layer by layer."""
    sims = []
    for idx in range(num_layers):
        op = os.path.join(hf_dir,     f"model.layers.{idx}.safetensors")
        fp = os.path.join(flyllm_dir, f"model.layers.{idx}.safetensors")
        if not os.path.exists(op) or not os.path.exists(fp):
            continue

        orig_t  = load_file(op, device="cpu")
        fly_all = load_file(fp, device="cpu")
        main    = [k for k in fly_all if not any(
            k.endswith(s) for s in [".__scales", ".__shape", ".__prec"]
        )]

        layer_sims = []
        for key in main:
            if key not in orig_t:
                continue
            q  = fly_all[key]
            sk = f"{key}.__scales"
            if sk in fly_all:
                meta = {"scales": fly_all[sk], "shape": fly_all[f"{key}.__shape"],
                        "prec": fly_all[f"{key}.__prec"]}
                t = dequantize_tensor(q, meta, DTYPE)
            else:
                t = q.to(DTYPE)
            layer_sims.append(cosine_sim(orig_t[key].to(DTYPE), t))

        if layer_sims:
            sims.append({"layer": idx, "cosine": float(np.mean(layer_sims))})

    return {
        "per_layer": sims,
        "avg":    round(float(np.mean([s["cosine"] for s in sims])), 6),
        "min":    round(float(np.min( [s["cosine"] for s in sims])), 6),
        "L0":     round(sims[0]["cosine"],  6) if sims else None,
        "L_last": round(sims[-1]["cosine"], 6) if sims else None,
    }


def run_pass(label, engine, tokenizer, cfg):
    """Run all prompts through engine, return results."""
    results = []
    for prompt, expected in zip(PROMPTS, EXPECTED):
        messages  = [{"role": "user", "content": prompt}]
        formatted = format_prompt(messages, cfg)
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"]

        t0     = time.time()
        tokens = list(engine.generate_tokens(
            input_ids, max_new_tokens=60, temperature=0.0
        ))
        elapsed = time.time() - t0
        answer  = tokenizer.decode(tokens, skip_special_tokens=True)
        correct = expected.lower() in answer.lower()

        results.append({
            "prompt":  prompt,
            "answer":  answer[:150],
            "time":    round(elapsed, 1),
            "tps":     round(60 / elapsed, 3),
            "correct": correct,
        })
        icon = "✅" if correct else "⚠️"
        print(f"  [{label}] {prompt[:45]}")
        print(f"           → {answer[:80]}")
        print(f"           {elapsed:.1f}s  {icon}\n")

    return {
        "label":    label,
        "results":  results,
        "avg_time": round(float(np.mean([r["time"] for r in results])), 1),
        "avg_tps":  round(float(np.mean([r["tps"]  for r in results])), 3),
        "accuracy": sum(r["correct"] for r in results),
    }


def run_benchmark(model_id: str) -> dict:
    from transformers import AutoTokenizer

    hf_dir     = get_hf_cache_dir(model_id)
    flyllm_dir = get_flyllm_dir(model_id)

    if not hf_dir:
        raise FileNotFoundError(
            f"Model not in HF cache. Run: flyllm run {model_id} --prompt 'test' first."
        )
    if not os.path.exists(os.path.join(flyllm_dir, "flyllm_meta.json")):
        raise FileNotFoundError(
            f"FlyLLM model not found. Run: flyllm quantize {model_id} first."
        )

    cfg       = load_config(model_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)

    print(f"\n{'='*65}")
    print(f"  FlyLLM Benchmark")
    print(f"  Model : {model_id}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*65}\n")

    report = {"model_id": model_id, "model_type": cfg.model_type, "device": DEVICE}

    # Pass 1 — Original float16
    print("[ 1/3 ] Original float16")
    eng_orig = MistralEngine(hf_dir, hf_dir, cfg, verbose=False)
    eng_orig.load()
    report["original"] = run_pass("float16", eng_orig, tokenizer, cfg)
    del eng_orig
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # Pass 2 — FlyLLM adaptive
    print("[ 2/3 ] FlyLLM Adaptive")
    eng_fly = MistralEngine(flyllm_dir, hf_dir, cfg, verbose=False)
    eng_fly.load()
    report["flyllm"] = run_pass("FlyLLM", eng_fly, tokenizer, cfg)
    del eng_fly
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # Pass 3 — Cosine similarity
    print("[ 3/3 ] Computing cosine similarity...")
    report["cosine"] = measure_cosine(hf_dir, flyllm_dir, cfg.num_hidden_layers)
    print(f"  avg={report['cosine']['avg']}  "
          f"L0={report['cosine']['L0']}  "
          f"L_last={report['cosine']['L_last']}\n")

    report["speedup"] = round(
        report["original"]["avg_time"] / max(report["flyllm"]["avg_time"], 0.1), 2
    )

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--output", default="benchmark/results.json")
    args = parser.parse_args()

    report = run_benchmark(args.model)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results → {args.output}")

    from benchmark.report import generate_report
    md      = generate_report(report)
    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Report  → {md_path}\n")
    print(md)


if __name__ == "__main__":
    main()
