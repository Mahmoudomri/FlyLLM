"""
FlyLLM - HuggingFace Fallback Engine
For any model not covered by custom engines (Phi, Qwen2, Falcon, etc.)

Lazy per-layer dequantization: the model stays compressed (int4/int8)
in RAM at all times. Each decoder layer is dequantized to fp16 only
during its own forward pass (via a pre-hook), and freed immediately
after (via a post-hook). At any given moment only ONE layer's worth
of fp16 weights exists on the GPU/CPU, instead of the whole model.

Trade-off: every layer is re-dequantized on every single forward call
(every token), which costs CPU/transfer time each step. This trades
speed for RAM/VRAM footprint — it does not make generation faster,
it makes it fit in less memory.
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

        # Build the model with EMPTY/meta weights only — no real fp16
        # storage is allocated for the decoder layers at this point.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=DTYPE,
            device_map="meta",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Keep compressed layers RAW in RAM — never dequantized here.
        # This dict is the only place the model's decoder weights live
        # persistently; it stays populated for the whole session.
        self.cache_layers = {}
        cfg = self.cfg
        for idx in range(cfg.num_hidden_layers):
            path = os.path.join(self.flyllm_dir, f"model.layers.{idx}.safetensors")
            if os.path.exists(path):
                self.cache_layers[idx] = load_file(path, device="cpu")
            if self.verbose:
                print(f"  Layer {idx:2d}/{cfg.num_hidden_layers-1} ✓ (compressed)",
                      end="\r", flush=True)

        if self.verbose:
            print(f"\n  All {cfg.num_hidden_layers} layers in RAM (compressed).")

        # Static weights (embed/norm/lm_head) need real values loaded
        # once — they're small relative to the decoder stack.
        self._load_static_weights()

        # Attach hooks: dequant-inject before each layer runs,
        # free right after it's done.
        self._attach_lazy_hooks()

        if self.verbose:
            print(f"  Model ready — layers stay compressed, "
                  f"decompressed on demand per forward call.")

    def _load_static_weights(self):
        for name in ["model.embed_tokens", "model.norm", "lm_head"]:
            path = os.path.join(self.hf_dir, f"{name}.safetensors")
            if not os.path.exists(path):
                continue
            d = load_file(path, device="cpu")
            for key, tensor in d.items():
                parts  = key.split(".")
                module = self.model
                try:
                    for p in parts[:-1]:
                        module = getattr(module, p)
                    param = getattr(module, parts[-1])
                    param.data = tensor.to(DTYPE).to(DEVICE)
                except AttributeError:
                    pass

    def _dequant_layer(self, idx):
        """Dequantize one layer's compressed tensors to fp16 on DEVICE.
        Reads only from self.cache_layers[idx] (RAM, compressed) —
        never mutates it, so the same compressed copy is reused
        every single token."""
        raw  = self.cache_layers.get(idx)
        if raw is None:
            return {}
        main = [k for k in raw if not any(
            k.endswith(s) for s in [".__scales", ".__shape", ".__prec"]
        )]
        out = {}
        for key in main:
            q  = raw[key]
            sk = f"{key}.__scales"
            if sk in raw:
                meta = {
                    "scales": raw[sk],
                    "shape":  raw[f"{key}.__shape"],
                    "prec":   raw[f"{key}.__prec"],
                }
                out[key] = dequantize_tensor(q, meta, DTYPE).to(DEVICE)
            else:
                # float16 layers (e.g. L0, last layer) have no meta —
                # already stored at full precision, just move+cast.
                out[key] = q.to(DTYPE).to(DEVICE)
        return out

    def _attach_lazy_hooks(self):
        cfg = self.cfg
        for idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[idx]

            def pre_hook(module, inputs, idx=idx, layer=layer):
                # Dequant just this layer, inject into its real params.
                weights = self._dequant_layer(idx)
                for key, tensor in weights.items():
                    short = key.replace(f"model.layers.{idx}.", "")
                    parts = short.split(".")
                    try:
                        sub = layer
                        for p in parts[:-1]:
                            sub = getattr(sub, p)
                        param = getattr(sub, parts[-1])
                        if isinstance(param, torch.nn.Parameter):
                            param.data = tensor
                        else:
                            setattr(sub, parts[-1], tensor)
                    except AttributeError:
                        pass
                return inputs

            def post_hook(module, inputs, output, idx=idx, layer=layer):
                # Free this layer's fp16 weights right after it's done —
                # the compressed original in self.cache_layers is untouched.
                for param in layer.parameters(recurse=True):
                    param.data = torch.empty(0, device=param.device, dtype=param.dtype)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                return output

            layer.register_forward_pre_hook(pre_hook)
            layer.register_forward_hook(post_hook)

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
