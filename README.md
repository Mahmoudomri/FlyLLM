# FlyLLM 🚀

**Adaptive quantization for local LLMs — smarter than uniform INT4.**

FlyLLM analyzes each layer using **3 mathematical metrics** and automatically assigns the optimal precision per layer. No calibration data. No manual configuration. One command.

```bash
pip install flyllm
flyllm run mistralai/Mistral-7B-v0.1 --prompt "What is AI?"
Why FlyLLM?

Every existing tool (AirLLM, GPTQ, Ollama) quantizes all layers the same way:

Layer 0  → INT4  ❌ loses outliers
Layer 15 → INT4
Layer 31 → INT4  ❌ loses critical information

FlyLLM analyzes layers first, then compresses adaptively:

Layer 0  → FP16 🔴 high sensitivity (outliers preserved)
Layer 1  → INT8 🟡 medium sensitivity
Layer 2  → INT4 🟢 low sensitivity
...
Layer 31 → FP16 🔴 high sensitivity
The 3 Metrics Used

FlyLLM computes a per-layer sensitivity score using:

Kurtosis → detects outlier-heavy distributions
Entropy → measures information density
Max Absolute Value → detects extreme activations
Final scoring:
Score = 0.5 × Kurtosis
      + 0.3 × Entropy
      + 0.2 × MaxAbs
Quantization Rules
Score ≥ 0.35 → FP16 (critical layers)
Score ≥ 0.15 → INT8  (moderate layers)
Score <  0.15 → INT4  (safe compression)
Benchmark Results — Mistral 7B
Metric	Float16	FlyLLM Adaptive
Avg time/prompt	145s	194s
vs uniform INT4	—	4.33x faster
Cosine similarity	1.000	0.9974
Accuracy (5 prompts)	5/5	5/5
Calibration data	Required	None
Quick Start
CLI
flyllm run mistralai/Mistral-7B-v0.1 --prompt "What is AI?"
flyllm chat mistralai/Mistral-7B-v0.1
flyllm profile mistralai/Mistral-7B-v0.1
flyllm quantize mistralai/Mistral-7B-v0.1
Python API
from flyllm import FlyLLM

model = FlyLLM.from_pretrained("mistralai/Mistral-7B-v0.1")

print(model.generate("What is AI?"))

for token in model.stream("Explain quantum computing"):
    print(token, end="", flush=True)

model.set_system("You are a helpful assistant.")
print(model.chat_turn("What is the capital of France?"))
How It Works
Load model from HuggingFace cache or download
Extract layer weights
Compute 3 metrics per layer:
Kurtosis
Entropy
Max Absolute Value
Compute sensitivity score
Assign precision per layer
Run inference fully in RAM
File Structure
flyllm/
├── config.py
├── profiler.py
├── quantizer.py
├── loader.py
├── chat.py
├── cli.py
└── engines/

benchmark/
├── run_benchmark.py
└── report.py
Supported Models
Model	Engine
Mistral 7B	Custom ⚡
Llama 3	Custom ⚡
Phi series	HF fallback
Qwen2	HF fallback
Any HF model	HF fallback
License

MIT