"""FlyLLM - Benchmark Report Generator"""

from datetime import datetime


def generate_report(data: dict) -> str:
    orig  = data["original"]
    fly   = data["flyllm"]
    cos   = data["cosine"]
    spdup = data["speedup"]

    lines = [
        f"## FlyLLM Benchmark — {data['model_type'].title()}",
        f"",
        f"*{datetime.now().strftime('%Y-%m-%d')} · Device: {data['device']}*",
        f"",
        f"### Speed",
        f"",
        f"| | Float16 (Original) | FlyLLM Adaptive |",
        f"|--|--|--|",
        f"| Avg time/prompt | {orig['avg_time']}s | {fly['avg_time']}s |",
        f"| Avg tok/s | {orig['avg_tps']} | {fly['avg_tps']} |",
        f"| **Speedup** | — | **{spdup}x** |",
        f"| Accuracy | {orig['accuracy']}/5 | {fly['accuracy']}/5 |",
        f"",
        f"### Weight Quality (Cosine Similarity vs Original)",
        f"",
        f"| Layer | FlyLLM |",
        f"|--|--|",
        f"| L0 — critical (first layer) | {cos['L0']} |",
        f"| Last layer — critical | {cos['L_last']} |",
        f"| Average all layers | {cos['avg']} |",
        f"| Minimum | {cos['min']} |",
        f"",
        f"### Per-Prompt Comparison",
        f"",
        f"| Prompt | Float16 | FlyLLM |",
        f"|--|--|--|",
    ]

    for o, f in zip(orig["results"], fly["results"]):
        oi = "✅" if o["correct"] else "⚠️"
        fi = "✅" if f["correct"] else "⚠️"
        lines.append(f"| {o['prompt'][:50]} | {oi} {o['time']}s | {fi} {f['time']}s |")

    lines += [
        f"",
        f"### How FlyLLM decides precision per layer",
        f"",
        f"| Metric | Variation on Mistral 7B | Weight in score |",
        f"|--|--|--|",
        f"| **Kurtosis** | 3.24 ← best | 50% |",
        f"| Entropy | 2.30 | 30% |",
        f"| Max Absolute Value | 1.27 | 20% |",
        f"| Std Dev | 0.07 ✗ | discarded |",
        f"| Norm L2 | 0.06 ✗ | discarded |",
        f"",
        f"> Score ≥ 0.35 → **float16** (critical)  ",
        f"> Score ≥ 0.15 → **INT8** (moderate)  ",
        f"> Score < 0.15 → **INT4** (compressible)  ",
        f"> Zero calibration data needed.",
    ]

    return "\n".join(lines)
