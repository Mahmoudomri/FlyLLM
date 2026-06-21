"""
FlyLLM - Llama 3 Engine
Same pattern as Mistral — different rope_theta and vocab size.
"""

import os
import math
import torch
import torch.nn.functional as F
from collections import OrderedDict
from safetensors.torch import load_file

from .base import BaseEngine
from ..config import ModelConfig
from ..quantizer import dequantize_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16

# Same speed/RAM knob as mistral.py -- see comments there.
HOT_CACHE_SIZE = 0


def _load_raw_layer(path):
    """Load one layer's tensors WITHOUT dequantizing — stays compressed in RAM."""
    return load_file(path, device="cpu")


def _dequantize_layer_to_device(raw, device):
    """Move ONE layer's compressed tensors to `device` and dequantize there.
    Batches the H2D transfer instead of doing one .to() call per tensor."""
    main = [k for k in raw if not any(
        k.endswith(s) for s in [".__scales", ".__shape", ".__prec"]
    )]

    keys_in_order = []
    cpu_tensors   = []
    for key in main:
        keys_in_order.append(("q", key))
        cpu_tensors.append(raw[key])
        sk = f"{key}.__scales"
        if sk in raw:
            keys_in_order.append(("scales", key))
            cpu_tensors.append(raw[sk])

    device_tensors = [t.to(device, non_blocking=True) for t in cpu_tensors]

    moved = {}
    for (kind, key), dt in zip(keys_in_order, device_tensors):
        moved.setdefault(key, {})[kind] = dt

    out = {}
    for key in main:
        q = moved[key]["q"]
        if "scales" not in moved[key]:
            out[key] = q.to(DTYPE)
            continue
        meta = {"scales": moved[key]["scales"],
                "shape": raw[f"{key}.__shape"], "prec": raw[f"{key}.__prec"]}
        out[key] = dequantize_tensor(q, meta, DTYPE)
    return out


class _HotCache:
    """Tiny LRU of decompressed layers -- see mistral.py for details."""
    def __init__(self, size):
        self.size = size
        self._od  = OrderedDict()

    def get_or_build(self, idx, raw, device):
        if self.size <= 0:
            return _dequantize_layer_to_device(raw, device)
        if idx in self._od:
            self._od.move_to_end(idx)
            return self._od[idx]
        W = _dequantize_layer_to_device(raw, device)
        self._od[idx] = W
        if len(self._od) > self.size:
            self._od.popitem(last=False)
        return W


def _build_rope(seq_len, head_dim, theta):
    pos   = torch.arange(seq_len, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang   = torch.outer(pos, freqs)
    return torch.cos(ang).to(DTYPE).to(DEVICE), torch.sin(ang).to(DTYPE).to(DEVICE)


def _apply_rope(x, cos, sin):
    d       = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return x * torch.cat([c,c],-1) + torch.cat([-x2,x1],-1) * torch.cat([s,s],-1)


def _rms_norm(x, weight, eps):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (weight * x * torch.rsqrt(v + eps)).to(DTYPE)


def _layer_fwd(h, W, cos, sin, cfg, pfx, layer_cache):
    NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
    HD, eps = cfg.hidden_size // NH, cfg.rms_norm_eps

    def w(n): return W[f"{pfx}.{n}"].to(DEVICE)
    B, S, _ = h.shape
    past_len = layer_cache["k"].shape[2] if layer_cache["k"] is not None else 0

    res = h
    x   = _rms_norm(h, w("input_layernorm.weight"), eps)
    Q   = F.linear(x, w("self_attn.q_proj.weight")).view(B,S,NH,HD).transpose(1,2)
    K   = F.linear(x, w("self_attn.k_proj.weight")).view(B,S,NKV,HD).transpose(1,2)
    V   = F.linear(x, w("self_attn.v_proj.weight")).view(B,S,NKV,HD).transpose(1,2)
    Q   = _apply_rope(Q, cos[past_len:past_len+S], sin[past_len:past_len+S])
    K   = _apply_rope(K, cos[past_len:past_len+S], sin[past_len:past_len+S])

    if layer_cache["k"] is not None:
        K = torch.cat([layer_cache["k"], K], dim=2)
        V = torch.cat([layer_cache["v"], V], dim=2)
    layer_cache["k"] = K
    layer_cache["v"] = V

    rep   = NH // NKV
    K_rep = K.repeat_interleave(rep, dim=1)
    V_rep = V.repeat_interleave(rep, dim=1)
    sc    = torch.matmul(Q, K_rep.transpose(-2,-1)) / math.sqrt(HD)

    if S > 1:
        mask = torch.full((S, S), float("-inf"), device=DEVICE).triu(1)
        if past_len > 0:
            mask = torch.cat([torch.zeros(S, past_len, device=DEVICE), mask], dim=1)
        sc = sc + mask.unsqueeze(0).unsqueeze(0)

    at  = F.softmax(sc.float(), -1).to(DTYPE)
    o   = torch.matmul(at,V_rep).transpose(1,2).contiguous().view(B,S,-1)
    h   = res + F.linear(o, w("self_attn.o_proj.weight"))

    res  = h
    x    = _rms_norm(h, w("post_attention_layernorm.weight"), eps)
    gate = F.linear(x, w("mlp.gate_proj.weight"))
    up   = F.linear(x, w("mlp.up_proj.weight"))
    h    = res + F.linear(F.silu(gate.float()).to(DTYPE) * up, w("mlp.down_proj.weight"))
    return h


class LlamaEngine(BaseEngine):

    def load(self):
        cfg = self.cfg
        if self.verbose:
            print(f"  Loading Llama {cfg.num_hidden_layers} layers into RAM...")

        self.cache["layers"] = {}
        for i in range(cfg.num_hidden_layers):
            path = os.path.join(self.flyllm_dir, f"model.layers.{i}.safetensors")
            if os.path.exists(path):
                self.cache["layers"][i] = _load_raw_layer(path)
            if self.verbose:
                print(f"  Layer {i:2d}/{cfg.num_hidden_layers-1} ✓", end="\r", flush=True)

        if self.verbose:
            print(f"  All {cfg.num_hidden_layers} layers in RAM (compressed).          ")

        def _load_hf(fname, key_filter=None):
            p = os.path.join(self.hf_dir, fname)
            if not os.path.exists(p): return None
            d = load_file(p, device="cpu")
            if key_filter:
                k = [x for x in d if key_filter in x][0]
                return d[k].to(DTYPE)
            return list(d.values())[0].to(DTYPE)

        self.cache["embed"]   = _load_hf("model.embed_tokens.safetensors", "embed_tokens")
        self.cache["norm"]    = _load_hf("model.norm.safetensors")
        self.cache["lm_head"] = _load_hf("lm_head.safetensors")

        HD = cfg.hidden_size // cfg.num_attention_heads
        self.cache["cos"], self.cache["sin"] = _build_rope(
            cfg.max_position_embeddings, HD, cfg.rope_theta
        )

        self._hot_cache = _HotCache(HOT_CACHE_SIZE)

    def reset_cache(self):
        self.kv_cache = [
            {"k": None, "v": None} for _ in range(self.cfg.num_hidden_layers)
        ]

    def forward(self, input_ids):
        cfg = self.cfg
        if not hasattr(self, "kv_cache"):
            self.reset_cache()
        if not hasattr(self, "_hot_cache"):
            self._hot_cache = _HotCache(HOT_CACHE_SIZE)

        h   = F.embedding(input_ids.to(DEVICE), self.cache["embed"].to(DEVICE))
        cos, sin = self.cache["cos"], self.cache["sin"]

        # empty_cache() restored: needed for stability on small GPUs.
        for i in range(cfg.num_hidden_layers):
            W = self._hot_cache.get_or_build(i, self.cache["layers"][i], DEVICE)
            h = _layer_fwd(h, W, cos, sin, cfg, f"model.layers.{i}", self.kv_cache[i])
            del W
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        v = h.float().pow(2).mean(-1, keepdim=True)
        h = (self.cache["norm"].to(DEVICE) * h * torch.rsqrt(v + cfg.rms_norm_eps)).to(DTYPE)
        return F.linear(h, self.cache["lm_head"].to(DEVICE))