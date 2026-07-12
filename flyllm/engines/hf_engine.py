"""
FlyLLM - HuggingFace Fallback Engine
For any model (Mistral,Llama,Phi, Qwen2, Falcon, etc.)
"""
import os
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from .base import BaseEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16


class HFEngine(BaseEngine):

    def load(self):
        if self.verbose:
            print(f"  Loading via HuggingFace fallback ({self.cfg.model_type})...")

      
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=DTYPE,
            device_map="meta",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

      
        self._fix_rotary_buffers()

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

  
        self._load_static_weights()
        self._attach_lazy_hooks()
        self._dequant_stream    = torch.cuda.Stream() if DEVICE == "cuda" else None
        self._prefetch_cache    = {}   # idx -> dequantized weights, ready to use
        self._prefetch_events   = {}   # idx -> cuda.Event signaling "dequant done"
        self.EMPTY_CACHE_EVERY = 8
        if self.verbose:
            print(f"  Model ready — layers stay compressed, "
                  f"decompressed on demand per forward call "
                  f"(pipelined prefetch: {'on' if self._dequant_stream else 'off'}).")

    def _fix_rotary_buffers(self):
    
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
                # Move compressed bytes + scales to GPU BEFORE dequanting —
                # this is the key change vs the CPU-side version.
                q_dev      = q.to(DEVICE, non_blocking=True)
                scales_dev = raw[sk].to(DEVICE, non_blocking=True).float()
                shape_data = raw[f"{key}.__shape"].tolist()
                prec       = int(raw[f"{key}.__prec"].item())
                orig_shape, pad = shape_data[:-1], shape_data[-1]

                if prec == 2:  # int4, nibble-packed
                    packed = q_dev  # uint8, half length
                    low    = (packed & 0x0F).to(torch.int16) - 8
                    high   = ((packed >> 4) & 0x0F).to(torch.int16) - 8
                    unpacked = torch.empty(
                        packed.numel() * 2, dtype=torch.int16, device=DEVICE
                    )
                    unpacked[0::2] = low
                    unpacked[1::2] = high
                    n_blocks = scales_dev.numel()
                    blocks   = unpacked.float().reshape(n_blocks, -1)
                else:  # int8
                    blocks = q_dev.float()

                flat = (blocks * scales_dev.unsqueeze(1)).flatten()
                if pad:
                    flat = flat[:-pad]
                out[key] = flat.reshape(orig_shape).to(DTYPE)
            else:
                # float16 layers (e.g. L0, last layer) have no meta —
                # already stored at full precision, just move+cast.
                out[key] = q.to(DTYPE).to(DEVICE)
        return out

    def _prefetch_layer(self, idx):
      
        if idx >= self.cfg.num_hidden_layers or idx in self._prefetch_cache:
            return
        if self._dequant_stream is None:
            # No CUDA streams available (CPU-only) — just do it inline.
            self._prefetch_cache[idx] = self._dequant_layer(idx)
            return
        with torch.cuda.stream(self._dequant_stream):
            self._prefetch_cache[idx] = self._dequant_layer(idx)
            event = torch.cuda.Event()
            event.record(self._dequant_stream)
            self._prefetch_events[idx] = event

    def _get_layer_weights(self, idx):
    
        if idx not in self._prefetch_cache:
            self._prefetch_layer(idx)
        if self._dequant_stream is not None and idx in self._prefetch_events:
           
            torch.cuda.current_stream().wait_event(self._prefetch_events.pop(idx))
        return self._prefetch_cache.pop(idx)

    def _attach_lazy_hooks(self):
        cfg = self.cfg
        for idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[idx]

            def pre_hook(module, inputs, idx=idx, layer=layer):
            
                weights = self._get_layer_weights(idx)

            
                if DEVICE == "cuda":
                    current = torch.cuda.current_stream()
                    for t in weights.values():
                        t.record_stream(current)

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

               
                self._prefetch_layer(idx + 1)
                return inputs

            def post_hook(module, inputs, output, idx=idx, layer=layer):
            
                for sub_module in layer.modules():
                    for pname, p in list(sub_module._parameters.items()):
                        if p is not None and p.numel() > 0:
                            empty = torch.empty(0, device=p.device, dtype=p.dtype)
                            sub_module._parameters[pname] = torch.nn.Parameter(
                                empty, requires_grad=False
                            )
                if DEVICE == "cuda" and (idx % self.EMPTY_CACHE_EVERY == 0):
                    torch.cuda.empty_cache()
                return output

            layer.register_forward_pre_hook(pre_hook)
            layer.register_forward_hook(post_hook)

    def reset_cache(self):
        self.past_key_values = None
        # Drop any leftover prefetch state from a previous generation.
        if hasattr(self, "_prefetch_cache"):
            self._prefetch_cache.clear()
        if hasattr(self, "_prefetch_events"):
            self._prefetch_events.clear()

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
