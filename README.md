# FlyLLM 🚀

Adaptive layer-wise quantization for local LLMs — smarter than uniform INT4.

FlyLLM analyzes each transformer layer using three statistical features and computes a single sensitivity score to automatically assign optimal precision per layer.

No calibration data. No manual tuning. One command.

pip install flyllm  
flyllm run mistralai/Mistral-7B-v0.1 --prompt "What is AI?"

# Why FlyLLM?

Most quantization tools (GPTQ, AWQ, Ollama, etc.) apply the same precision across all layers:

Layer 0 → INT4 ❌ loses outliers  
Layer 15 → INT4  
Layer 31 → INT4 ❌ loses critical information  

FlyLLM adapts precision per layer:

Layer 0 → FP16 🔴 score 0.42 (Kurtosis: 19.5, Entropy: 2.30, MaxAbs: 1.27)  
Layer 1 → INT8 🟡 score 0.21 (Kurtosis: 7.8, Entropy: 1.95, MaxAbs: 0.88)  
Layer 2 → INT4 🟢 score 0.08 (Kurtosis: 4.5, Entropy: 1.10, MaxAbs: 0.42)  
Layer 31 → FP16 🔴 score 0.39 (Kurtosis: 9.9, Entropy: 2.10, MaxAbs: 1.05)  

# How It Works

Score formula: Score = 0.5 × Kurtosis + 0.3 × Entropy + 0.2 × MaxAbs

Decision rules:
≥ 0.35 → FP16  
≥ 0.15 → INT8  
< 0.15 → INT4  

# Benchmark Results — Mistral 7B

| Metric | Float16 | FlyLLM |
|--------|--------|--------|
| Avg time/prompt | 145s | 194s |
| Cosine similarity | 1.000 | 0.9974 |
| Accuracy (5 prompts) | 5/5 | 5/5 |
| Calibration data | Required | None |

# Quick Start

CLI:
flyllm run mistralai/Mistral-7B-v0.1 --prompt "What is AI?"  
flyllm chat mistralai/Mistral-7B-v0.1  
flyllm profile mistralai/Mistral-7B-v0.1  
flyllm quantize mistralai/Mistral-7B-v0.1  

Python:
from flyllm import FlyLLM  
model = FlyLLM.from_pretrained("mistralai/Mistral-7B-v0.1")  
print(model.generate("What is AI?"))  

for token in model.stream("Explain quantum computing"):  
    print(token, end="", flush=True)  

model.set_system("You are a helpful assistant.")  
print(model.chat_turn("What is the capital of France?"))

# How It Works Pipeline

1. Load model from HuggingFace cache or download  
2. Profile layers using Kurtosis, Entropy, Max Absolute Value  
3. Compute sensitivity score  
4. Assign precision per layer  
5. Run inference fully in RAM  

# File Structure

flyllm/
├── config.py
├── profiler.py
├── quantizer.py
├── loader.py
├── chat.py
├── cli.py
└── engines/
    ├── base.py
    ├── mistral.py
    ├── llama.py
    └── hf_engine.py

benchmark/
├── run_benchmark.py
└── report.py

# Supported Models

Mistral 7B → Custom ⚡  
Llama 3 → Custom ⚡  
Phi series → HF fallback  
Qwen2 → HF fallback  
Any HF model → HF fallback  

# License

MIT