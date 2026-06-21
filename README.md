# FlyLLM 🚀

**Adaptive quantization for local LLMs — smarter than uniform INT4.**

FlyLLM analyzes each layer using **3 metrics** and automatically picks the best precision per layer. No calibration data. No manual config. One command.

```bash
pip install flyllm
flyllm run mistralai/Mistral-7B-v0.1 --prompt "What is AI?"
```

---

## Why FlyLLM?

Most tools quantize every layer the same way, which can damage sensitive layers.

FlyLLM analyzes first, then compresses:

```
score = 0.5·kurtosis + 0.3·entropy + 0.2·max_abs   (normalized per layer)

Layer 0  → float16  🔴  score 0.81 — outliers protected
Layer 1  → INT8     🟡  score 0.22 — moderate sensitivity
Layer 2  → INT4     🟢  score 0.09 — safe to compress
...
Layer 31 → float16  🔴  score 0.68 — outliers protected
```

---



See [Run Benchmark](#run-benchmark) below to generate real numbers for your model.

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

**1. Check cache** — FlyLLM checks if the model is already downloaded locally. If not, it downloads it.

**2. Check compressed model** — FlyLLM checks `~/flyllmmodel/ModelName/`. If not found, it profiles and quantizes.

**3. Profile** — Three metrics per layer, combined into one sensitivity score:

| Metric | What it measures | Weight |
|--|--|--|
| **Kurtosis** | Outlier strength in weight distribution | 50% |
| Entropy | Information density / randomness | 30% |
| Max Absolute Value | Largest weight magnitude | 20% |

Score decides precision:
- ≥ 0.35 → **float16** (critical layers)
- ≥ 0.15 → **INT8** (moderate)
- < 0.15 → **INT4** (safe to compress)

**4. Quantize** — Only the layer files are saved to `~/flyllmmodel/ModelName/`. Static weights (embed, norm, lm_head) stay in the original cache — never duplicated.

**5. Load into RAM once** — All 32 compressed layers are dequantized into RAM. Static weights are read from cache. Token generation runs entirely from RAM — no disk reads during inference.

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
├── model.embed_tokens.safetensors
├── model.norm.safetensors
├── lm_head.safetensors
└── model.layers.{0-31}.safetensors  ← used for profiling only
```

---

## Project Structure

```
flyllm/
├── config.py       Model detection + chat templates + cache paths
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
| Mistral 7B | HF fallback ⚡ |
| Llama 3 8B/70B | HF fallback⚡ |
| Phi-2, Phi-3 | HF fallback |
| Qwen2 | HF fallback |
| Any HF transformer | HF fallback |

---

## License
MIT
