"""
FlyLLM - Public API
Handles the full flow:
1. Check HF cache — download if needed (via AirLLM split)
2. Check ~/flyllmmodel/ModelName — quantize if needed
3. Load compressed layers into RAM
4. Inference
"""

import os
import json
from typing import Optional, Iterator
from transformers import AutoTokenizer

from .config import (load_config, format_prompt, ModelConfig,
                     get_hf_cache_dir, get_flyllm_dir, get_model_name)
from .profiler import profile_model
from .quantizer import quantize_model
from .engines import get_engine


def _split_model_layers(snapshot_dir: str, verbose: bool = True) -> str:
    """
    Split a downloaded HF snapshot into per-layer safetensors files.
    Works whether the original checkpoint was saved as a single
    model.safetensors file or as multiple sharded files (with or
    without model.safetensors.index.json) — unlike AirLLM's splitter,
    which hard-requires an index.json and fails on single-file models.
    """
    import re
    from collections import defaultdict
    from safetensors import safe_open
    from safetensors.torch import save_file

    split_dir = os.path.join(snapshot_dir, "splitted_model")
    os.makedirs(split_dir, exist_ok=True)

    layer0_marker = os.path.join(split_dir, "model.layers.0.safetensors")
    if os.path.exists(layer0_marker):
        return split_dir

    # Figure out which safetensors shard(s) exist for this snapshot.
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        single_file = os.path.join(snapshot_dir, "model.safetensors")
        if not os.path.exists(single_file):
            raise RuntimeError(
                f"No model.safetensors or model.safetensors.index.json found in "
                f"{snapshot_dir}. This model may use a .bin checkpoint format, "
                "which isn't supported yet."
            )
        shard_files = ["model.safetensors"]

    if verbose:
        print(f"  Splitting {len(shard_files)} shard(s) into per-layer files...")

    layer_pat = re.compile(r"model\.layers\.(\d+)\.")
    layer_tensors = defaultdict(dict)
    static_tensors = {}

    for shard_name in shard_files:
        shard_path = os.path.join(snapshot_dir, shard_name)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                m = layer_pat.search(key)
                if m:
                    layer_tensors[int(m.group(1))][key] = tensor
                else:
                    static_tensors[key] = tensor

    for idx, tensors in layer_tensors.items():
        out_path = os.path.join(split_dir, f"model.layers.{idx}.safetensors")
        save_file(tensors, out_path)

    if static_tensors:
        save_file(static_tensors, os.path.join(split_dir, "static.safetensors"))

    if verbose:
        print(f"  ✅ Split {len(layer_tensors)} layers → {split_dir}")

    return split_dir


def _ensure_hf_model(model_id: str, verbose: bool = True) -> str:
    """
    Make sure the model is downloaded and split in HF cache.
    Returns path to splitted_model directory.
    """
    # Check if already in cache
    cached = get_hf_cache_dir(model_id)
    if cached:
        if verbose:
            print(f"  ✅ Found in HF cache: {cached}")
        return cached

    if verbose:
        print(f"  Downloading {model_id} from HuggingFace...\n")

    from huggingface_hub import snapshot_download
    snapshot_dir = snapshot_download(model_id)

    split_dir = _split_model_layers(snapshot_dir, verbose=verbose)

    cached = get_hf_cache_dir(model_id)
    if cached:
        return cached
    return split_dir


def _ensure_flyllm_model(model_id: str, hf_dir: str,
                          verbose: bool = True) -> str:
    """
    Make sure the compressed FlyLLM model exists.
    Returns path to ~/flyllmmodel/ModelName directory.
    """
    flyllm_dir = get_flyllm_dir(model_id)
    meta_path  = os.path.join(flyllm_dir, "flyllm_meta.json")

    if os.path.exists(meta_path):
        # Check if all layer files exist
        with open(meta_path) as f:
            meta = json.load(f)
        layer0 = os.path.join(flyllm_dir, "model.layers.0.safetensors")
        if os.path.exists(layer0):
            if verbose:
                print(f"  ✅ Found compressed model: {flyllm_dir}")
                print(f"     {meta['avg_bits']} bits avg  |  "
                      f"{meta['size_reduction']}% smaller  |  "
                      f"{meta['float16_layers']} float16  "
                      f"{meta['int8_layers']} int8  "
                      f"{meta['int4_layers']} int4\n")
            return flyllm_dir

    # Profile + quantize
    if verbose:
        print(f"\n  Profiling layers...\n")

    cfg         = load_config(model_id)
    profile     = profile_model(
        model_id,
        hf_dir=hf_dir,
        output_path=os.path.join(flyllm_dir, "flyllm_profile.json"),
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Quantizing layers...\n")

    quantize_model(
        model_id=model_id,
        hf_dir=hf_dir,
        profile=profile,
        output_dir=flyllm_dir,
        verbose=verbose,
    )

    return flyllm_dir


class FlyLLM:
    """
    Main FlyLLM interface.

    # Fully automatic — checks cache, downloads if needed, quantizes if needed
    model = FlyLLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    print(model.generate("What is AI?"))

    # Load already-compressed model directly
    model = FlyLLM.load("~/flyllmmodel/Mistral-7B-v0.1")
    """

    def __init__(self, engine, cfg: ModelConfig, tokenizer):
        self.engine    = engine
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self._history  = []
        self._system   = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        verbose:  bool = True,
    ) -> "FlyLLM":
        """
        Full automatic pipeline:
        1. Check HF cache → download if needed
        2. Check ~/flyllmmodel → profile + quantize if needed
        3. Load into RAM and return
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  FlyLLM — {model_id}")
            print(f"{'='*60}\n")

        # Step 1 — HF cache
        hf_dir = _ensure_hf_model(model_id, verbose=verbose)

        # Step 2 — FlyLLM compressed model
        flyllm_dir = _ensure_flyllm_model(model_id, hf_dir, verbose=verbose)

        # Step 3 — Load
        return cls._load(model_id, flyllm_dir, hf_dir, verbose=verbose)

    @classmethod
    def load(cls, flyllm_dir: str, verbose: bool = True) -> "FlyLLM":
        """Load directly from a flyllm model directory."""
        flyllm_dir = os.path.expanduser(flyllm_dir)
        meta_path  = os.path.join(flyllm_dir, "flyllm_meta.json")

        if not os.path.exists(meta_path):
            raise ValueError(
                f"{flyllm_dir} is not a FlyLLM model directory. "
                "Run FlyLLM.from_pretrained() first."
            )

        with open(meta_path) as f:
            meta = json.load(f)

        model_id = meta["model_id"]
        hf_dir   = meta["hf_dir"]

        if not os.path.isdir(hf_dir):
            raise FileNotFoundError(
                f"Original HF model not found at: {hf_dir}\n"
                "The static weights (embed, norm, lm_head) are needed from the original."
            )

        if verbose:
            print(f"\n  Loading FlyLLM — {model_id}")
            print(f"  {meta['avg_bits']} bits avg  |  "
                  f"{meta['size_reduction']}% smaller  |  "
                  f"{meta['float16_layers']} float16  "
                  f"{meta['int8_layers']} int8  "
                  f"{meta['int4_layers']} int4\n")

        return cls._load(model_id, flyllm_dir, hf_dir, verbose=verbose)

    @classmethod
    def _load(cls, model_id, flyllm_dir, hf_dir, verbose=True):
        cfg       = load_config(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        engine    = get_engine(flyllm_dir, hf_dir, cfg, verbose=verbose)
        return cls(engine, cfg, tokenizer)

    # ── Generation ────────────────────────────────────────────

    def generate(
        self,
        prompt:         str,
        system:         Optional[str] = None,
        max_new_tokens: int   = 512,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
    ) -> str:
        messages  = [{"role": "user", "content": prompt}]
        formatted = format_prompt(messages, self.cfg, system or self._system)
        input_ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"]
        stop_ids = self.tokenizer("[INST]", add_special_tokens=False)["input_ids"]

        tokens    = list(self.engine.generate_tokens(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            tokenizer=self.tokenizer,
            stop_ids=stop_ids,
        ))
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def stream(
        self,
        prompt:         str,
        system:         Optional[str] = None,
        max_new_tokens: int   = 512,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
    ) -> Iterator[str]:
        messages  = [{"role": "user", "content": prompt}]
        formatted = format_prompt(messages, self.cfg, system or self._system)
        input_ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"]
        for token_id in self.engine.generate_tokens(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, stream=True,
            tokenizer=self.tokenizer,
        ):
            yield self.tokenizer.decode([token_id], skip_special_tokens=True)

    def chat_turn(self, user_msg: str, max_new_tokens: int = 512,
                  temperature: float = 0.7) -> str:
        self._history.append({"role": "user", "content": user_msg})
        formatted = format_prompt(self._history, self.cfg, self._system)
        input_ids = self.tokenizer(formatted, return_tensors="pt")["input_ids"]
        tokens    = list(self.engine.generate_tokens(
            input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            tokenizer=self.tokenizer,
        ))
        response  = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self._history.append({"role": "assistant", "content": response})
        return response

    def set_system(self, system: str):
        self._system = system

    def reset_history(self):
        self._history = []

    def __repr__(self):
        return f"FlyLLM(model={self.cfg.model_type}, layers={self.cfg.num_hidden_layers})"