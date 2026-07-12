# FlyLLM 🚀 — Precision Where It Counts

**Sensitivity-Aware Mixed-Precision Quantization for Memory-Efficient LLM Inference .**

FlyLLM profiles every layer of a model with three zero-calibration metrics
and assigns each one its own precision (float16 / int8 / int4), instead of
compressing the whole model uniformly. Layers stay compressed in RAM at all
times; only one layer at a time is decompressed on demand — the goal is
running large language model on machines with as little as ~4GB VRAM / ~16GB RAM.

```bash
pip install flyllm
flyllm run Qwen/Qwen2.5-7B-Instruct --prompt "What is AI?"
```

This is a memory-footprint tool, not a speed tool yet — see
[How it performs](#how-it-performs) for the honest picture.

---

## Why FlyLLM?

Most quantization tools compress every layer the same way. But layers
aren't equally sensitive — a handful of layers (typically the first and
last) carry extreme activation outliers that get destroyed by aggressive
low-bit quantization, while most of the rest of the model tolerates int4
just fine.

FlyLLM scores every layer, clusters the scores with k-means, and assigns
precision per layer automatically — no fixed thresholds, no calibration
dataset, no manual config:

```
score = 0.5·kurtosis_norm + 0.3·entropy_norm + 0.2·max_abs_norm

L0   → float16  🔴  extreme outlier layer — always protected
L1   → int8     🟡  elevated sensitivity — found by clustering, not forced
L2   → int4     🟢  flat baseline — safe to compress hard
...
L27  → float16  🔴  extreme outlier layer — always protected
```

The first and last layers are **always** float16, and are excluded from
the k-means computation itself — not just from the final assignment. 
This matters: these layers are typically more sensitive because the first
layer influences how input representations are formed and propagated through
the network, while the final layer directly affects output token prediction.
Leaving them in the clustering data introduces these high-sensitivity outliers,
which drags the median/MAD and compresses the effective score range for every 
other layer, making it much harder for k-means to find real structure among the middle layers.
Excluding them gives materially better resolution — allowing the adaptive algorithm to allocate 
precision more accurately where it is needed.


---

## Quick Start

### CLI

```bash
# Auto-download, profile, quantize, run — one command
flyllm run mistralai/Mistral-7B-Instruct-v0.3 --prompt "What is AI?"

# Interactive chat
flyllm chat  mistralai/Mistral-7B-Instruct-v0.3

# With system prompt
flyllm chat  mistralai/Mistral-7B-Instruct-v0.3 --system "You are a Python expert."

# Profile only — see layer analysis without quantizing
flyllm profile mistralai/Mistral-7B-Instruct-v0.3

# Quantize only
flyllm quantize mistralai/Mistral-7B-Instruct-v0.3
```

### Python API

```python
from flyllm import FlyLLM

# Full pipeline — downloads, profiles, quantizes, and loads
model = FlyLLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

print(model.generate(
    "A train leaves at 3pm going 60mph, another leaves at 4pm going 90mph "
    "on the same track. When does the second train catch up?",
    max_new_tokens=200,
    temperature=0.0,   # greedy — deterministic, recommended for correctness checks
))

# Stream tokens
for token in model.stream("Write a haiku about compressed neural networks"):
    print(token, end="", flush=True)

# Multi-turn conversation
model.set_system("You are a terse, precise research assistant.")
model.chat_turn("What's the difference between int8 and int4 quantization?")
model.chat_turn("Which one loses more accuracy, and why?")  # remembers context

# Load an already-quantized model directly, skip the from_pretrained checks
model = FlyLLM.load("~/flyllmmodel/Qwen2.5-7B-Instruct")
```

---

## How it works

**1. Check cache** — downloads the model from HuggingFace if not already
cached, and splits it into per-layer safetensors files.

**2. Check compressed model** — looks under `~/flyllmmodel/ModelName/`. If
missing, profiles and quantizes automatically.

**3. Profile** (`profiler.py`) — three metrics per layer, zero calibration
data:

| Metric | What it captures | Weight |
|---|---|---|
| Kurtosis | Extreme outlier activations — destroyed by low-bit quantization | 50% |
| Entropy | Information density of the weight distribution | 30% |
| Max Absolute Value | Dynamic range of the weights | 20% |

Precision is decided by **1D k-means (k=3)** on the score column, run
separately for every model — not a fixed SCORE cutoff. The first and last
layers are excluded from that clustering and forced to float16 directly
(see below). `k=3` isn't a tuning knob; it's simply the number of
bit-width tiers FlyLLM offers.

**4. Quantize** (`quantizer.py`) — float16 layers are stored as-is; int8
layers use per-block (block size 32) symmetric quantization; int4 layers
use the same scheme with **nibble packing** — two int4 values packed into
a single byte, which is what actually delivers int4's ~4x size reduction
vs fp16 on disk. Only layer files are written to `~/flyllmmodel/ModelName/`;
static weights (embed, norm, lm_head) stay in the HF cache, never
duplicated.

**5. Load & generate** (`engines/hf_engine.py`, the fallback used for any
architecture without a dedicated custom engine — Phi, Qwen2, Falcon,
etc.):

- The model is built with `device_map="meta"` — no real weight storage
  allocated for the decoder stack at load time.
- All layers are loaded from disk **compressed** into a CPU RAM dict and
  **stay there for the whole session** — this is the only place the
  model's weights live persistently.
- A forward **pre-hook** on each decoder layer transfers that layer's
  compressed bytes to GPU and dequantizes them there (not on CPU), then
  injects the resulting fp16 tensors into the layer's parameters right
  before it runs.
- A forward **post-hook** frees that layer's fp16 weights immediately
  after, replacing them with empty tensors.
- **Pipelined prefetch**: while the GPU computes layer N's forward matmuls
  on the main CUDA stream, layer N+1's dequant runs concurrently on a
  second stream, hiding dequant cost behind compute. `record_stream()` is
  called on consumed tensors to stop PyTorch's caching allocator from
  reusing their memory across streams before the main stream is done
  reading them — without this, the failure mode is silent corruption
  (NaNs in the logits), not a clean crash.

At any given moment, at most ~2 layers' worth of fp16 weights exist in
memory — never the whole model.

---



## How it performs

Real numbers, FlyLLM vs. unquantized fp16 on identical prompts, same
hardware:

| Model | Layers | fp16/int8/int4 | Avg bits | Size reduction | Peak VRAM | Speed |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 28 | 4 / 6 / 18 | 6.57 |  57.4% | 2.13 GB | 0.40 tok/s |
| Mistral-7B-Instruct-v0.3 | 32 | 3 / 20 / 9 | 7.62 | 52.3% | 2.21 GB | ~0.45 tok/s |
| Phi-3-mini-4k-instruct | 32 | 2 / 13 / 17 | 6.38 | 60.2% | 2.36 GB | ~0.80 tok/s |

Plain fp16 needs roughly 14GB for Mistral-7B and 7.6GB for Phi-3-mini —
FlyLLM's VRAM numbers above are a large cut versus either. Generation
quality matched the fp16 baseline on identical prompts (math word
problems, factual QA, one-sentence explanations) in the large majority of
tested cases, with one documented exception fixed by edge exclusion
(above).

Every layer is re-decompressed on every single token, not once at load
time — so generation is slower than fp16 running fully in VRAM, and
slower than dedicated engines like llama.cpp/Ollama, which use hand-written
low-bit compute kernels instead of decompress-then-matmul. A Triton kernel
that computes directly on packed, mixed-precision weights is in progress
on a separate branch, aimed at closing that gap.

---

## Project structure

```
flyllm/
├── config.py            Model detection, chat templates, cache paths
├── profiler.py           Kurtosis + Entropy + MaxAbs layer analysis, k-means tiering
├── quantizer.py            Per-layer compression, int4 nibble packing
├── loader.py              FlyLLM.from_pretrained() / FlyLLM.load()
├── chat.py                Interactive streaming chat
├── cli.py                  flyllm run / chat / quantize / profile
└── engines/
    ├── base.py              Shared token generation loop
    ├── hf_engine.py           Active engine: lazy per-layer dequant + CUDA pipelining
    └── ...                     Experimental hand-written engines (not in active use)

benchmark/    fp16 vs FlyLLM comparison + report generator
tests/        unit tests for config, profiler, quantizer, engine, API
examples/     runnable usage scripts
```

## Requirements

```
torch>=2.0.0
transformers>=4.35.0
safetensors>=0.4.0
huggingface_hub>=0.19.0
numpy>=1.24.0
```

A CUDA GPU is strongly recommended — CPU-only works but loses every
GPU-side optimization this project relies on.

## License

MIT — see [LICENSE](LICENSE).
