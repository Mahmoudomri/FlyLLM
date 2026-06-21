"""
FlyLLM - Mistral Engine
Proven engine from flyllm_engine.py v3.
Loads compressed layers from flyllm_dir into RAM.
Reads embed/norm/lm_head from hf_dir (original HF cache).
"""

import os
import math
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .base import BaseEngine
from ..config import ModelConfig
from ..quantizer import dequantize_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16


# ── Layer loading — stays compressed in RAM ───────────────────

def _load_raw_layer(path: str) -> dict:
    """Load one layer's tensors WITHOUT dequantizing — stays int4/int8 in RAM."""
    return load_file(path, device="cpu")


def _dequantize_layer_to_device(raw: dict, device) -> dict:
    """
    Move ONE layer's still-compressed tensors to `device` and dequantize
    them there. Called fresh every forward pass — RAM only ever holds the
    compressed (~5GB) version, never the full fp16 (~14GB) version.
    """
    main = [k for k in raw if not any(
        k.endswith(s) for s in [".__scales", ".__shape", ".__prec"]
    )]
    out = {}
    for key in main:
        q  = raw[key].to(device, non_blocking=True)
        sk = f"{key}.__scales"
        if sk not in raw:
            out[key] = q.to(DTYPE)
            continue
        meta = {
            "scales": raw[sk].to(device, non_blocking=True),
            "shape":  raw[f"{key}.__shape"],   # tiny int32 list, fine on CPU
            "prec":   raw[f"{key}.__prec"],
        }
        out[key] = dequantize_tensor(q, meta, DTYPE)
    return out


# ── RoPE ─────────────────────────────────────────────────────

def _build_rope(seq_len: int, head_dim: int, theta: float):
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


# ── Layer forward — KV-cache aware ────────────────────────────
# h is ONLY the new tokens for this step (full prompt on the first
# call, exactly 1 token on every call after that). Past K/V live in
# layer_cache and get reused instead of recomputed every step.

def _layer_fwd(h, W, cos, sin, cfg: ModelConfig, pfx: str, layer_cache: dict):
    NH  = cfg.num_attention_heads
    NKV = cfg.num_key_value_heads
    HD  = cfg.hidden_size // NH
    eps = cfg.rms_norm_eps

    def w(n): return W[f"{pfx}.{n}"].to(DEVICE)

    B, S, _ = h.shape  # S = number of NEW tokens this call (not total so far)
    past_len = layer_cache["k"].shape[2] if layer_cache["k"] is not None else 0

    # Attention
    res = h
    x   = _rms_norm(h, w("input_layernorm.weight"), eps)
    Q   = F.linear(x, w("self_attn.q_proj.weight")).view(B,S,NH,HD).transpose(1,2)
    K   = F.linear(x, w("self_attn.k_proj.weight")).view(B,S,NKV,HD).transpose(1,2)
    V   = F.linear(x, w("self_attn.v_proj.weight")).view(B,S,NKV,HD).transpose(1,2)

    # RoPE uses absolute position (past_len .. past_len+S), not 0..S
    Q = _apply_rope(Q, cos[past_len:past_len+S], sin[past_len:past_len+S])
    K = _apply_rope(K, cos[past_len:past_len+S], sin[past_len:past_len+S])

    # Append this step's K/V to the running cache instead of recomputing old ones
    if layer_cache["k"] is not None:
        K = torch.cat([layer_cache["k"], K], dim=2)
        V = torch.cat([layer_cache["v"], V], dim=2)
    layer_cache["k"] = K
    layer_cache["v"] = V

    rep   = NH // NKV
    K_rep = K.repeat_interleave(rep, dim=1)
    V_rep = V.repeat_interleave(rep, dim=1)

    sc = torch.matmul(Q, K_rep.transpose(-2,-1)) / math.sqrt(HD)  # (B,NH,S,total_len)

    if S > 1:
        # Prefill: causal mask over the new chunk, full visibility into any past
        total_len = K_rep.shape[2]
        mask = torch.full((S, S), float("-inf"), device=DEVICE).triu(1)
        if past_len > 0:
            mask = torch.cat(
                [torch.zeros(S, past_len, device=DEVICE), mask], dim=1
            )
        sc = sc + mask.unsqueeze(0).unsqueeze(0)
    # S == 1 (decode step): the single new token may attend to everything
    # already cached — no mask needed, causality is automatic.

    at = F.softmax(sc.float(), -1).to(DTYPE)
    o  = torch.matmul(at, V_rep).transpose(1,2).contiguous().view(B,S,-1)
    h  = res + F.linear(o, w("self_attn.o_proj.weight"))

    # MLP SwiGLU
    res  = h
    x    = _rms_norm(h, w("post_attention_layernorm.weight"), eps)
    gate = F.linear(x, w("mlp.gate_proj.weight"))
    up   = F.linear(x, w("mlp.up_proj.weight"))
    h    = res + F.linear(
        F.silu(gate.float()).to(DTYPE) * up,
        w("mlp.down_proj.weight")
    )
    return h


class MistralEngine(BaseEngine):

    def load(self):
        cfg = self.cfg

        if self.verbose:
            print(f"  Loading {cfg.num_hidden_layers} compressed layers into RAM "
                  f"(staying compressed — decompressed per-layer at use time)...")

        # ── Load compressed layers from flyllm_dir — RAW, not dequantized ──
        self.cache["layers"] = {}
        for i in range(cfg.num_hidden_layers):
            path = os.path.join(self.flyllm_dir, f"model.layers.{i}.safetensors")
            if os.path.exists(path):
                self.cache["layers"][i] = _load_raw_layer(path)
            if self.verbose:
                print(f"  Layer {i:2d}/{cfg.num_hidden_layers-1} ✓", end="\r", flush=True)

        if self.verbose:
            print(f"  All {cfg.num_hidden_layers} layers in RAM (compressed).          ")

        # ── Load static weights from HF cache (original) ─────
        def _load_hf(fname, key_filter=None):
            p = os.path.join(self.hf_dir, fname)
            if not os.path.exists(p):
                return None
            d = load_file(p, device="cpu")
            if key_filter:
                k = [x for x in d if key_filter in x][0]
                return d[k].to(DTYPE)
            return list(d.values())[0].to(DTYPE)

        self.cache["embed"]   = _load_hf("model.embed_tokens.safetensors", "embed_tokens")
        self.cache["norm"]    = _load_hf("model.norm.safetensors")
        self.cache["lm_head"] = _load_hf("lm_head.safetensors")

        if self.verbose:
            print(f"  Static weights loaded from HF cache.")

        # ── Build RoPE cache ──────────────────────────────────
        if self.verbose:
            print(f"  Building RoPE cache (seq_len={cfg.max_position_embeddings})...")
        HD = cfg.hidden_size // cfg.num_attention_heads
        self.cache["cos"], self.cache["sin"] = _build_rope(
            cfg.max_position_embeddings, HD, cfg.rope_theta
        )
        if self.verbose:
            print(f"  RoPE cache ready.")

    def reset_cache(self):
        """Call at the start of each generation — wipes the per-layer K/V cache."""
        self.kv_cache = [
            {"k": None, "v": None} for _ in range(self.cfg.num_hidden_layers)
        ]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: full prompt on the FIRST call (prefill), exactly the
        single newest token on every call after that (decode). The KV
        cache (built in reset_cache) carries everything already seen,
        so we never recompute the whole sequence from scratch.
        """
        cfg = self.cfg
        cos = self.cache["cos"]
        sin = self.cache["sin"]

        if not hasattr(self, "kv_cache"):
            self.reset_cache()

        # Embedding (only for the new tokens passed in)
        h = F.embedding(
            input_ids.to(DEVICE),
            self.cache["embed"].to(DEVICE)
        )

        # 32 layers — each one decompressed fresh, right as it hits the GPU
        for i in range(cfg.num_hidden_layers):
            W = _dequantize_layer_to_device(self.cache["layers"][i], DEVICE)
            h = _layer_fwd(h, W, cos, sin, cfg, f"model.layers.{i}", self.kv_cache[i])
            del W
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # Final norm
        v = h.float().pow(2).mean(-1, keepdim=True)
        h = (self.cache["norm"].to(DEVICE) * h *
             torch.rsqrt(v + cfg.rms_norm_eps)).to(DTYPE)

        # LM head
        return F.linear(h, self.cache["lm_head"].to(DEVICE))
