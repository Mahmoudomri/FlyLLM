"""
FlyLLM - HuggingFace Fallback Engine
For any model not covered by custom engines (Phi, Qwen2, Falcon, etc.)
Uses HF's own forward pass with dequantized weights.
"""

import os
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from .base import BaseEngine
from ..quantizer import dequantize_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16


class HFEngine(BaseEngine):

    def load(self):
        if self.verbose:
            print(f"  Loading via HuggingFace fallback ({self.cfg.model_type})...")

        # IMPORTANT: self.hf_dir is the *split* directory (per-layer
        # safetensors only) — it has no config.json/tokenizer, so HF
        # can't build the model class from it. Load the architecture
        # from the original model id/repo instead, then inject our
        # dequantized layer weights on top.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else "cpu",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Inject our dequantized weights
        self._inject_weights()

    def _inject_weights(self):
        cfg = self.cfg
        for idx in range(cfg.num_hidden_layers):
            path = os.path.join(self.flyllm_dir, f"model.layers.{idx}.safetensors")
            if not os.path.exists(path):
                continue

            all_t = load_file(path, device="cpu")
            main  = [k for k in all_t if not any(
                k.endswith(s) for s in [".__scales", ".__shape", ".__prec"]
            )]
            layer = self.model.model.layers[idx]

            for key in main:
                q  = all_t[key]
                sk = f"{key}.__scales"
                if sk in all_t:
                    meta   = {"scales": all_t[sk], "shape": all_t[f"{key}.__shape"],
                              "prec": all_t[f"{key}.__prec"]}
                    tensor = dequantize_tensor(q, meta, DTYPE)
                else:
                    tensor = q.to(DTYPE)

                short = key.replace(f"model.layers.{idx}.", "")
                parts = short.split(".")
                try:
                    module = layer
                    for part in parts[:-1]:
                        module = getattr(module, part)
                    param = getattr(module, parts[-1])
                    if isinstance(param, torch.nn.Parameter):
                        param.data = tensor.to(param.device)
                except AttributeError:
                    pass

    def reset_cache(self):
        self.past_key_values = None

    def forward(self, input_ids):
        if not hasattr(self, "past_key_values"):
            self.reset_cache()
        with torch.no_grad():
            out = self.model(
                input_ids.to(DEVICE),
                past_key_values=self.past_key_values,
                use_cache=True,
            )
        self.past_key_values = out.past_key_values
        return out.logits