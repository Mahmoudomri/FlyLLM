# FlyLLM 🚀

**Adaptive quantization for local LLMs — smarter than uniform INT4.**

FlyLLM analyzes each layer using **3 mathematical metrics** and automatically assigns the optimal precision per layer. No calibration data. No manual configuration. One command.

```bash
pip install flyllm
flyllm run mistralai/Mistral-7B-v0.1 --prompt "What is AI?"
```

---

## Why FlyLLM?

Every existing tool (AirLLM, GPTQ, Ollama) quantizes all layers identically:

```
Layer 0  → INT4  ❌  critical layer — outliers lost
Layer 15 → INT4
Layer 31 → INT4  ❌  critical layer — outliers lost
```

FlyLLM analyzes first, then compresses:

```
Layer 0  → float16  🔴  kurtosis 19.5 — outliers protected
Layer 1  → INT8     🟡  kurtosis 7.8  — moderate sensitivity
Layer 2  → INT4     🟢  kurtosis 4.5  — safe to compress
...
Layer 31 → float16  🔴  kurtosis 9.9  — outliers protected
```

---

## Benchmark Results — Mistral 7B

| | Float16 | FlyLLM Adaptive |
|--|--|--|
| Avg time/prompt | 145s | 194s |
| vs AirLLM uniform INT4 | — | **4.33x faster** |
| L0 cosine (critical) | 1.000 | **1.000** ✅ |
| L31 cosine (critical) | 1.000 | **1.000** ✅ |
| Avg cosine similarity | 1.000 | **0.9974** |
| Accuracy (5 prompts) | 5/5 | 5/5 |
| Calibration data | — | ❌ none needed |

---

## Quick Start

### CLI

```bash
# Auto-download, profile, quantize, run — one command
flyllm run mistralai/Mistral-7B-v0.1 --prompt "What is AI?"

# Interactive chat
flyllm chat mistralai/Mistral-7B-v0.1

# With system prompt
flyllm chat mistralai/Mistral-7B-v0.1 --system "You are a Python expert."

# Profile only — see layer analysis
flyllm profile mistralai/Mistral-7B-v0.1

# Quantize only
flyllm quantize mistralai/Mistral-7B-v0.1
```

### Python API

```python
from flyllm import FlyLLM

# Full pipeline — one line
model = FlyLLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Generate
print(model.generate("What is AI?"))

# With system prompt
print(model.generate(
    "Write a Python sort function",
    system="You are a Python expert. Be concise.",
))

# Stream tokens
for token in model.stream("Explain quantum computing"):
    print(token, end="", flush=True)

# Multi-turn conversation
model.set_system("You are a helpful assistant.")
r1 = model.chat_turn("What is the capital of France?")
r2 = model.chat_turn("What is its population?")  # remembers context

# Load already-quantized model directly
model = FlyLLM.load("~/flyllmmodel/Mistral-7B-v0.1")
```

---

## How It Works

### Step 1 — Check HF cache
FlyLLM checks if the model is already in `~/.cache/huggingface/`. If not, downloads and splits it via AirLLM.

### Step 2 — Check compressed model
FlyLLM checks `~/flyllmmodel/ModelName/`. If not found, runs the profiler and quantizer.

### Step 3 — Profile (if needed)

Three metrics per layer, combined into one sensitivity score:

| Metric | Variation on Mistral 7B | Weight |
|--|--|--|
| **Kurtosis** | 3.24 ← winner | 50% |
| Entropy | 2.30 | 30% |
| Max Absolute Value | 1.27 | 20% |

Score decides precision:
- ≥ 0.35 → **float16** (critical layers)
- ≥ 0.15 → **INT8** (moderate)
- < 0.15 → **INT4** (safe to compress)

### Step 4 — Quantize (if needed)
Only `model.layers.{idx}.safetensors` files are saved to `~/flyllmmodel/ModelName/`. Static weights (embed, norm, lm_head) stay in HF cache — never duplicated.

### Step 5 — Load all layers into RAM once
All 32 compressed layers dequantized into RAM. Static weights read from HF cache. Token generation runs entirely from RAM — no SSD reads during inference.

---

## File Structure

```
~/flyllmmodel/Mistral-7B-v0.1/     ← compressed layers only
├── model.layers.0.safetensors      🔴 float16
├── model.layers.1.safetensors      🟡 int8
├── model.layers.2.safetensors      🟢 int4
├── ...
├── model.layers.31.safetensors     🔴 float16
├── flyllm_profile.json
└── flyllm_meta.json

~/.cache/huggingface/.../           ← original weights (untouched)
├── model.embed_tokens.safetensors  ← read at inference
├── model.norm.safetensors          ← read at inference
├── lm_head.safetensors             ← read at inference
└── model.layers.{0-31}.safetensors ← used for profiling only
```

---

## Project Structure

```
flyllm/
├── config.py       Auto model detection + chat templates + cache paths
├── profiler.py     Kurtosis + Entropy + MaxAbs layer analysis
├── quantizer.py    Adaptive per-layer compression
├── loader.py       FlyLLM.from_pretrained() + FlyLLM.load()
├── chat.py         Interactive streaming chat
├── cli.py          flyllm run / chat / quantize / profile
└── engines/
    ├── base.py     Token generation loop
    ├── mistral.py  Custom Mistral engine ⚡ proven
    ├── llama.py    Custom Llama 3 engine ⚡
    └── hf_engine.py  HuggingFace fallback (any model)

benchmark/
├── run_benchmark.py  float16 vs FlyLLM comparison
└── report.py         Markdown report generator
```

---

## Run Benchmark

```bash
python -m benchmark.run_benchmark --model mistralai/Mistral-7B-v0.1
```

Outputs `benchmark/results.json` and `benchmark/results.md`.

---

## Supported Models

| Model | Engine |
|--|--|
| Mistral 7B | Custom ⚡ |
| Llama 3 8B/70B | Custom ⚡ |
| Phi-2, Phi-3 | HF fallback |
| Qwen2 | HF fallback |
| Any HF transformer | HF fallback |

---

## License
MIT
