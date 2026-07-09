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

Pipelined prefetch: while the GPU is busy running layer N's forward
matmuls (main CUDA stream), layer N+1's dequant runs concurrently on
a SEPARATE CUDA stream. By the time layer N finishes, N+1's weights
are often already sitting in fp16 ready to use — so the dequant cost
is hidden behind the compute cost instead of adding to it. On CPU-only
setups there are no streams, so this degrades gracefully to inline
(non-overlapped) dequant.

Trade-off: every layer is still re-dequantized on every single forward
call (every token) — that repeated work doesn't go away, and RAM/VRAM
footprint stays low (at most ~2 layers' worth of fp16 weights exist at
once: the one currently running + the one being prefetched). What the
pipeline buys back is wall-clock time, by overlapping dequant with
compute instead of doing them strictly back to back.
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

        # Pipelining state: a second CUDA stream lets us dequant layer
        # N+1 WHILE the GPU is still busy computing layer N's matmuls,
        # instead of doing dequant-then-compute strictly back to back.
        # On CPU-only setups there's no stream concept, so this is a
        # no-op and behaves like the non-pipelined version.
        self._dequant_stream    = torch.cuda.Stream() if DEVICE == "cuda" else None
        self._prefetch_cache    = {}   # idx -> dequantized weights, ready to use
        self._prefetch_events   = {}   # idx -> cuda.Event signaling "dequant done"

        # torch.cuda.empty_cache() forces a full device sync — calling
        # it on EVERY layer defeats the pipeline (it blocks until the
        # background dequant stream is fully caught up, which is
        # exactly the wait we're trying to hide). Instead, only free
        # every EMPTY_CACHE_EVERY layers — frequent enough to avoid
        # fragmentation OOMs on small GPUs, rare enough to preserve
        # most of the overlap. Set to 1 to restore the old per-layer
        # behavior if you hit memory issues.
        self.EMPTY_CACHE_EVERY = 8

        if self.verbose:
            print(f"  Model ready — layers stay compressed, "
                  f"decompressed on demand per forward call "
                  f"(pipelined prefetch: {'on' if self._dequant_stream else 'off'}).")

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
        """Dequantize one layer's compressed tensors to fp16, with the
        actual dequant MATH happening on DEVICE (GPU), not on CPU.

        Old approach: dequant on CPU (slow Python/PyTorch-CPU tensor
        ops), then transfer the big fp16 result to GPU.
        New approach: transfer the small COMPRESSED bytes to GPU first
        (cheap — int4 is ~4x smaller than fp16), then do the unpack +
        scale multiply on GPU (CUDA tensor ops are much faster than
        the same ops on CPU). Net effect: less data moved, and the
        compute itself is faster.

        Reads only from self.cache_layers[idx] (RAM, compressed) —
        never mutates it, so the same compressed copy is reused
        every single token.
        """
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
        """Launch dequant of layer `idx` on the background stream, so
        it runs concurrently with whatever the main stream is doing
        (i.e. the current layer's forward matmuls). Result lands in
        self._prefetch_cache[idx] once done; an Event lets pre_hook
        wait for exactly that point instead of a full device sync."""
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
        """Fetch layer idx's dequantized weights, waiting only if the
        background prefetch for it hasn't finished yet. Falls back to
        an inline (blocking) dequant if it was never prefetched at all
        — e.g. layer 0 on the very first forward call."""
        if idx not in self._prefetch_cache:
            self._prefetch_layer(idx)
        if self._dequant_stream is not None and idx in self._prefetch_events:
            # Make the MAIN stream wait for the dequant stream to reach
            # this point — cheap if it's already done, blocks briefly
            # if the layer's forward pass was faster than its dequant.
            torch.cuda.current_stream().wait_event(self._prefetch_events.pop(idx))
        return self._prefetch_cache.pop(idx)

    def _attach_lazy_hooks(self):
        cfg = self.cfg
        for idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[idx]

            def pre_hook(module, inputs, idx=idx, layer=layer):
                # Grab this layer's weights — dequant may already be
                # done in the background (prefetched during the
                # PREVIOUS layer's forward pass), or done inline now
                # if this is the very first layer.
                weights = self._get_layer_weights(idx)
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

                # Kick off dequant of the NEXT layer now, on the
                # background stream, so it overlaps with THIS layer's
                # upcoming forward compute instead of waiting for it.
                self._prefetch_layer(idx + 1)
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
