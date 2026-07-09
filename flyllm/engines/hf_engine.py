"""
FlyLLM - HuggingFace Fallback Engine
For any model not covered by custom engines (Phi, Qwen2, Falcon, etc.)

Lazy per-layer dequantization: the model stays compressed (int4/int8)
in RAM at all times. Each decoder layer is dequantized to fp16 only
during its own forward pass (via a pre-hook), and freed immediately
after (via a post-hook). At any given moment only ONE layer's worth
of fp16 weights exists on the GPU/CPU, instead of the whole model.

NOTE: the model is built with device_map="meta", so its parameters
start as meta tensors (no real storage). Meta tensors are a distinct
type from real tensors, so PyTorch refuses `param.data = real_tensor`
(RuntimeError: incompatible tensor type). The fix is to replace the
Parameter object itself in the module's `_parameters` dict instead of
mutating `.data` on the existing (meta) Parameter.

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

        # RoPE's inv_freq buffer is computed from config, not loaded
        # from the checkpoint — but under device_map="meta" it's still
        # created as a meta tensor with no real data. Fix it now,
        # before anything tries to use it.
        self._fix_rotary_buffers()

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
        # once — they're small relative to the decoder stack. These
        # also start as meta tensors and need the same _parameters
        # replacement trick, not a .data assignment.
        self._load_static_weights()

        # Attach hooks: dequant-inject before each layer runs,
        # free right after it's done.
        self._attach_lazy_hooks()

        if self.verbose:
            print(f"  Model ready — layers stay compressed, "
                  f"decompressed on demand per forward call.")

    def _fix_rotary_buffers(self):
        """inv_freq is computed from config (not loaded from checkpoint),
        but under device_map='meta' it's created as a meta tensor too and
        never gets real values assigned. Recompute it directly from the
        standard RoPE formula and replace the buffer object (same trick
        as _set_param, to avoid the meta-tensor copy error)."""
        cfg = self.cfg
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        base = cfg.rope_theta
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        for name, _ in list(self.model.named_buffers()):
            if name.endswith("inv_freq"):
                module = self.model
                parts = name.split(".")
                for p in parts[:-1]:
                    module = getattr(module, p)
                module._buffers[parts[-1]] = inv_freq.to(DEVICE)

    def _set_param(self, module, name, tensor):
        """Replace a (possibly meta) Parameter with a real one, or set
        a plain buffer/attribute. Works regardless of whether the
        current value is a meta tensor or a real one."""
        if name in module._parameters:
            module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
        elif name in module._buffers:
            module._buffers[name] = tensor
        else:
            setattr(module, name, tensor)

    def _load_static_weights(self):
        # embed_tokens, norm, and lm_head are all saved together in a
        # single static.safetensors file by the splitter (loader.py),
        # not as three separate files — this must match that layout.
        path = os.path.join(self.hf_dir, "static.safetensors")
        if not os.path.exists(path):
            if self.verbose:
                print(f"  WARNING: static.safetensors not found at {path} — "
                      f"embed/norm/lm_head will stay uninitialized (meta)!")
            return
        d = load_file(path, device="cpu")
        for key, tensor in d.items():
            parts  = key.split(".")
            module = self.model
            try:
                for p in parts[:-1]:
                    module = getattr(module, p)
                self._set_param(module, parts[-1], tensor.to(DTYPE).to(DEVICE))
            except AttributeError:
                if self.verbose:
                    print(f"  WARNING: could not set static param {key}")

    def _dequant_layer(self, idx):
        """Dequantize one layer's compressed tensors to fp16 on DEVICE.
        Reads only from self.cache_layers[idx] (RAM, compressed) —
        never mutates it, so the same compressed copy is reused
        every single token."""
        raw = self.cache_layers.get(idx)
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
                        self._set_param(sub, parts[-1], tensor)
                    except AttributeError:
                        pass
                return inputs

            def post_hook(module, inputs, output, idx=idx, layer=layer):
                # Free this layer's fp16 weights right after it's done —
                # the compressed original in self.cache_layers is untouched.
                # Replace each real Parameter with a tiny empty one (not
                # meta — meta tensors can trigger the same set_data
                # incompatibility on the NEXT pre_hook injection).
                for sub_module in layer.modules():
                    for pname, p in list(sub_module._parameters.items()):
                        if p is not None and p.numel() > 0:
                            empty = torch.empty(0, device=p.device, dtype=p.dtype)
                            sub_module._parameters[pname] = torch.nn.Parameter(
                                empty, requires_grad=False
                            )
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
